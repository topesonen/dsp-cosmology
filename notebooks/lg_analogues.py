"""Local Group analogue utilities for IllustrisTNG group catalogs.

This module contains:
- Selection of central + largest satellite (by stellar mass) per FoF group
- Pair finding within a distance window in a periodic box
- Isolation filtering: reject pairs with a much more massive third object in the same FoF group(s)

Assumptions:
- TNG group catalogs loaded via illustris_python (il.groupcat.loadSubhalos/loadHalos/loadHeader)
- Snapshot is typically z=0 (snap=99 for TNG300-1), but the utilities are general
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from scipy.spatial import cKDTree
import illustris_python as il


@dataclass(frozen=True)
class Sample:
    """A filtered subhalo sample and associated arrays.

    keep_idx is the global subhalo index into the full Subhalo table.
    pos is in ckpc/h, vel in km/s.
    mstar and mdm are in Msun.
    """
    keep_idx: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    mtype: np.ndarray
    grnr: np.ndarray
    mstar: np.ndarray
    mdm: np.ndarray


@dataclass(frozen=True)
class PairSet:
    """Pair indices and derived kinematics.

    i, j are indices into the provided sample arrays.
    dist_kpc is physical separation in kpc (assuming z=0 or treating ckpc as kpc).
    v_r and v_t are in km/s.
    """
    i: np.ndarray
    j: np.ndarray
    dist_kpc: np.ndarray
    v_r: np.ndarray
    v_t: np.ndarray
    same_host: np.ndarray


def load_header_constants(basePath: str, snap: int) -> Dict[str, float]:
    """Load commonly used header constants.

    Returns:
        dict with keys:
        - h: HubbleParam
        - box_ckpch: BoxSize (ckpc/h)
    """
    header = il.groupcat.loadHeader(basePath, snap)
    return {
        "h": float(header["HubbleParam"]),
        "box_ckpch": float(header["BoxSize"]),
    }


def select_central_plus_largest_satellite_by_stellar_mass(
    sub: Dict[str, Any],
    basePath: str,
    snap: int,
    h: float,
    mstar_min: float,
    mstar_max: float,
) -> Sample:
    """Select central + largest satellite (by stellar mass) per FoF group.

    Logic:
    - Define stellar mass from SubhaloMassType[:,4] (stars+wind) in Msun.
    - Keep FoF groups whose central subhalo is in the stellar-mass window.
    - For each selected group, also keep the satellite with the largest stellar mass
      among satellites that are also inside the stellar-mass window.
    - Exclude SubhaloFlag != 1 and invalid SubhaloGrNr.

    Args:
        sub: dict returned by il.groupcat.loadSubhalos(...).
        basePath: simulation outputs path (contains group catalog).
        snap: snapshot number.
        h: Hubble parameter (little h).
        mstar_min: minimum stellar mass (Msun).
        mstar_max: maximum stellar mass (Msun).

    Returns:
        Sample containing global indices and sliced arrays.
    """
    halo_cat = il.groupcat.loadHalos(basePath, snap, fields=["GroupFirstSub", "GroupNsubs"])
    first = halo_cat["GroupFirstSub"]

    # SubhaloMassType units are 1e10 Msun/h; index 4 is stars+wind.
    mstar_all = sub["SubhaloMassType"][:, 4].astype(np.float64) * 1e10 / h
    grnr_all = sub["SubhaloGrNr"].astype(np.int64)

    good = (sub["SubhaloFlag"] == 1) & (grnr_all >= 0) & (mstar_all > 0)
    in_window = (mstar_all >= mstar_min) & (mstar_all <= mstar_max)

    is_central = np.zeros(sub["count"], dtype=bool)
    central_idx_all = first[first >= 0]
    is_central[central_idx_all] = True

    # Centrals inside the window define the set of eligible FoF groups.
    central_keep = is_central & good & in_window
    central_idx = np.nonzero(central_keep)[0]

    groups_selected = np.unique(grnr_all[central_idx])

    # Satellites inside the window, restricted to the selected groups.
    sat_mask = good & in_window & (~is_central) & np.isin(grnr_all, groups_selected)
    sat_idx = np.nonzero(sat_mask)[0]

    if sat_idx.size == 0:
        keep_idx = central_idx
    else:
        sat_gr = grnr_all[sat_idx]
        sat_m = mstar_all[sat_idx]

        # Sort by group id ascending, then by stellar mass descending.
        order = np.lexsort((-sat_m, sat_gr))
        sat_idx_s = sat_idx[order]
        sat_gr_s = sat_gr[order]

        # First satellite per group in this ordering is the largest by stellar mass.
        new_group = np.r_[True, sat_gr_s[1:] != sat_gr_s[:-1]]
        best_sat_idx = sat_idx_s[new_group]

        keep_idx = np.unique(np.concatenate([central_idx, best_sat_idx]))

    mtype = sub["SubhaloMassType"][keep_idx]
    mstar = mtype[:, 4].astype(np.float64) * 1e10 / h
    mdm = mtype[:, 1].astype(np.float64) * 1e10 / h

    return Sample(
        keep_idx=keep_idx,
        pos=sub["SubhaloPos"][keep_idx],
        vel=sub["SubhaloVel"][keep_idx],
        mtype=mtype,
        grnr=sub["SubhaloGrNr"][keep_idx],
        mstar=mstar,
        mdm=mdm,
    )


def find_pairs_periodic(
    pos: np.ndarray,
    vel: np.ndarray,
    grnr: np.ndarray,
    h: float,
    box_ckpch: float,
    r_min_kpc: float,
    r_max_kpc: float,
) -> PairSet:
    """Find pairs within [r_min_kpc, r_max_kpc] using a periodic cKDTree.

    Args:
        pos: positions (ckpc/h), shape (N,3)
        vel: velocities (km/s), shape (N,3)
        grnr: FoF host group index per object, shape (N,)
        h: Hubble parameter (little h)
        box_ckpch: periodic box size in ckpc/h
        r_min_kpc: minimum separation in kpc (physical)
        r_max_kpc: maximum separation in kpc (physical)

    Returns:
        PairSet with indices into the input arrays and derived kinematics.
    """
    r_min = r_min_kpc * h
    r_max = r_max_kpc * h

    tree = cKDTree(pos, boxsize=box_ckpch)
    pairs = np.array(list(tree.query_pairs(r_max)), dtype=np.int64)

    if pairs.size == 0:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        empty_b = np.array([], dtype=bool)
        return PairSet(i=empty_i, j=empty_i, dist_kpc=empty_f, v_r=empty_f, v_t=empty_f, same_host=empty_b)

    i = pairs[:, 0]
    j = pairs[:, 1]

    # Minimum-image convention for periodic displacement vector.
    dr = pos[j] - pos[i]
    dr -= box_ckpch * np.round(dr / box_ckpch)

    dist = np.linalg.norm(dr, axis=1)
    keep = dist >= r_min

    i = i[keep]
    j = j[keep]
    dr = dr[keep]
    dist = dist[keep]

    dist_kpc = dist / h

    dv = vel[j] - vel[i]
    rhat = dr / dist[:, None]
    v_r = np.einsum("ij,ij->i", dv, rhat)

    v2 = np.einsum("ij,ij->i", dv, dv)
    v_t = np.sqrt(np.maximum(0.0, v2 - v_r * v_r))

    same_host = (grnr[i] == grnr[j])

    return PairSet(i=i, j=j, dist_kpc=dist_kpc, v_r=v_r, v_t=v_t, same_host=same_host)


def isolation_filter_no_third_factor_x_in_same_group(
    sub: Dict[str, Any],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    sample_grnr: np.ndarray,
    sample_keep_idx_global: np.ndarray,
    h: float,
    x: float,
) -> np.ndarray:
    """Isolation filter: reject pairs with a very massive third object in the same FoF group(s).

    For each pair (a,b), reject if there exists a third subhalo in the relevant FoF group(s)
    whose stellar mass >= x * max(Mstar[a], Mstar[b]).

    The third-object search is done on all subhalos with SubhaloFlag==1 and Mstar>0
    but restricted to the FoF groups that appear in at least one pair member.

    Args:
        sub: dict returned by il.groupcat.loadSubhalos(...).
        pair_i, pair_j: pair indices into the sample arrays.
        sample_grnr: FoF group index per sample object.
        sample_keep_idx_global: global Subhalo table indices per sample object.
        h: Hubble parameter (little h).
        x: mass factor threshold.

    Returns:
        Boolean mask, True for pairs that pass isolation.
    """
    mstar_all = sub["SubhaloMassType"][:, 4].astype(np.float64) * 1e10 / h
    grnr_all = sub["SubhaloGrNr"].astype(np.int64)
    good_all = (sub["SubhaloFlag"] == 1) & (grnr_all >= 0) & (mstar_all > 0)

    groups_in_pairs = np.unique(
        np.concatenate([sample_grnr[pair_i], sample_grnr[pair_j]]).astype(np.int64)
    )

    cand = good_all & np.isin(grnr_all, groups_in_pairs)

    cand_idx = np.nonzero(cand)[0]
    cand_gr = grnr_all[cand_idx]
    cand_m = mstar_all[cand_idx]

    # Sort by group id ascending, then mass descending.
    order = np.lexsort((-cand_m, cand_gr))
    cand_idx = cand_idx[order]
    cand_gr = cand_gr[order]
    cand_m = cand_m[order]

    # Build top-3 lookup per group to find the largest possible third object quickly.
    starts = np.flatnonzero(np.r_[True, cand_gr[1:] != cand_gr[:-1]])
    ends = np.r_[starts[1:], cand_gr.size]

    top_idx: Dict[int, np.ndarray] = {}
    top_m: Dict[int, np.ndarray] = {}

    for s, e in zip(starts, ends):
        g = int(cand_gr[s])
        take = min(3, e - s)
        top_idx[g] = cand_idx[s:s + take]
        top_m[g] = cand_m[s:s + take]

    # Stellar masses for the sample objects (Msun).
    mstar_sample = sub["SubhaloMassType"][sample_keep_idx_global, 4].astype(np.float64) * 1e10 / h

    keep_iso = np.ones(pair_i.size, dtype=bool)

    for k in range(pair_i.size):
        a = int(pair_i[k])
        b = int(pair_j[k])

        ga = int(sample_grnr[a])
        gb = int(sample_grnr[b])

        ma = float(mstar_sample[a])
        mb = float(mstar_sample[b])
        scale = max(ma, mb)

        a_global = int(sample_keep_idx_global[a])
        b_global = int(sample_keep_idx_global[b])

        def violates(g: int) -> bool:
            if g not in top_idx:
                return False
            idxs = top_idx[g]
            ms = top_m[g]
            for idx_val, m_val in zip(idxs, ms):
                idx_val = int(idx_val)
                if (idx_val != a_global) and (idx_val != b_global):
                    return float(m_val) >= x * scale
            return False

        if ga == gb:
            if violates(ga):
                keep_iso[k] = False
        else:
            if violates(ga) or violates(gb):
                keep_iso[k] = False

    return keep_iso


def isolation_filter_no_massive_intruder_within_distance_x(
    sub: Dict[str, Any],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    sample_pos: np.ndarray,
    sample_keep_idx_global: np.ndarray,
    h: float,
    box_ckpch: float,
    r_iso_kpc: float = 2000.0,
    mass_factor: float = 0.5
) -> np.ndarray:
    """
    Isolation filter: reject pairs if any third subhalo within r_iso_kpc 
    of the pair's Center of Mass is more massive than the pair's combined mass.

    Args:
        sub: dict returned by il.groupcat.loadSubhalos(...).
        pair_i, pair_j: pair indices into the sample arrays.
        sample_pos: Positions (ckpc/h) of the subhalos in the sample.
        sample_keep_idx_global: global Subhalo table indices per sample object.
        h: Hubble parameter (little h).
        box_ckpch: Periodic box size in ckpc/h.
        r_iso_kpc: Isolation radius in physical kpc (default 2000/2Mpc).

    Returns:
        Boolean mask, True for pairs that pass isolation.
    """

    m_all = sub["SubhaloMassType"][:, 4].astype(np.float64) * 1e10 / h
    pos_all = sub["SubhaloPos"].astype(np.float64)
    pos_all %= box_ckpch

    idx_i = sample_keep_idx_global[pair_i]
    idx_j = sample_keep_idx_global[pair_j]
    
    # Filter for valid subhalos
    valid = (sub["SubhaloFlag"] == 1) & (m_all > 0)
    tree = cKDTree(pos_all[valid], boxsize=box_ckpch)
    valid_indices = np.where(valid)[0]
    m_valid = m_all[valid]

    # Total mass of each pair member
    m_i = sub["SubhaloMass"][idx_i] * 1e10 / h
    m_j = sub["SubhaloMass"][idx_j] * 1e10 / h
    m_combined = mass_factor*(m_i + m_j)
    
    # Calculate Center of Mass
    pos_i = sample_pos[pair_i] % box_ckpch
    pos_j = sample_pos[pair_j] % box_ckpch
    
    dr = pos_j - pos_i
    dr -= box_ckpch * np.round(dr / box_ckpch) # Shortest path
    com = pos_i + (dr * (m_j / m_combined)[:, None]) # Mass-weighted CoM
    com %= box_ckpch # Wrap back into box

    # Perform spatial search
    r_iso_ckpch = r_iso_kpc * h
    keep_iso = np.ones(len(pair_i), dtype=bool)

    for k in range(len(pair_i)):
        # Find all subhalos within 2 Mpc of CoM
        neighbor_indices = tree.query_ball_point(com[k], r_iso_ckpch)
        
        if not neighbor_indices:
            continue
            
        # Get global indices of neighbors and their masses
        actual_neighbor_globals = valid_indices[neighbor_indices]
        neighbor_masses = m_valid[neighbor_indices]
        
        # Check if any neighbor is heavier than combined mass
        for m_neigh, idx_neigh in zip(neighbor_masses, actual_neighbor_globals):
            if idx_neigh != idx_i[k] and idx_neigh != idx_j[k]:
                if m_neigh > m_combined[k]:
                    keep_iso[k] = False
                    break
                    
    return keep_iso


def calculate_force_ratio(
    sub: Dict[str, Any],
    pair_set: PairSet,
    sample: Sample,
    h: float,
    box_ckpch: float,
    search_radius_kpc: float = 2000.0,
) -> np.ndarray:
    """
    Calculate the maximum ratio of external tidal force 
    to internal gravitational force for each galaxy pair.

    Args:
        sub: dict returned by il.groupcat.loadSubhalos(...).
        pair_set: PairSet object containing pair indices (i, j) and separations.
        sample: Sample object containing subhalo positions and global indices.
        h: Hubble parameter (little h).
        box_ckpch: Periodic box size in ckpc/h.
        search_radius_kpc: Radius to search for external perturbers (default 2000/2Mpc).

    Returns:
        np.ndarray: Force ratio F for each pair
    """
   
    m_all = sub["SubhaloMass"].astype(np.float64) * 1e10 / h
    pos_all = (sub["SubhaloPos"].astype(np.float64)) % box_ckpch
    
    # Filter valid subhalos
    valid_mask = (sub["SubhaloFlag"] == 1) & (m_all > 0)
    pos_valid = pos_all[valid_mask]
    m_valid = m_all[valid_mask]
    global_indices_valid = np.where(valid_mask)[0]
    
    # Find neighbors
    tree = cKDTree(pos_valid, boxsize=box_ckpch)
    search_r = search_radius_kpc * h
    neighbors = tree.query_ball_point(sample.pos % box_ckpch, r=search_r)
    
    # Find top two external forces and their source IDs
    max_f1 = np.zeros(len(sample.pos))    # Strongest force
    id_f1 = np.full(len(sample.pos), -1)  # ID of strongest
    max_f2 = np.zeros(len(sample.pos))    # Second strongest force

    softening_sq = 1.0 # Prevent 1/r^2 divergence
    
    for s_idx in range(len(sample.pos)):
        if not neighbors[s_idx]:
            continue
            
        n_idx_in_valid = neighbors[s_idx]
        n_global_ids = global_indices_valid[n_idx_in_valid]
        
        # Exclude the subhalo itself
        self_mask = n_global_ids != sample.keep_idx[s_idx]
        if not np.any(self_mask): continue

        dr = pos_valid[n_idx_in_valid][self_mask] - (sample.pos[s_idx] % box_ckpch)
        dr -= box_ckpch * np.round(dr / box_ckpch)
        dist_sq = np.sum(dr**2, axis=1) + softening_sq
        
        all_forces = m_valid[n_idx_in_valid][self_mask] / dist_sq
        
        # Sort forces to get top two
        if len(all_forces) >= 2:
            sort_idx = np.argsort(all_forces)[-2:]
            max_f1[s_idx] = all_forces[sort_idx[1]]
            id_f1[s_idx] = n_global_ids[self_mask][sort_idx[1]]
            max_f2[s_idx] = all_forces[sort_idx[0]]
        elif len(all_forces) == 1:
            max_f1[s_idx] = all_forces[0]
            id_f1[s_idx] = n_global_ids[self_mask][0]

    # Final pairs
    idx1, idx2 = pair_set.i, pair_set.j
    global_id1 = sample.keep_idx[idx1]
    global_id2 = sample.keep_idx[idx2]
    
    # Internal forces
    r_pair_sq = (pair_set.dist_kpc * h)**2 + softening_sq
    f_int_on_1 = m_all[global_id2] / r_pair_sq
    f_int_on_2 = m_all[global_id1] / r_pair_sq
    
    # Select max external force,
    f_ext_on_1 = np.where(id_f1[idx1] == global_id2, max_f2[idx1], max_f1[idx1])
    f_ext_on_2 = np.where(id_f1[idx2] == global_id1, max_f2[idx2], max_f1[idx2])
    
    ratio1 = f_ext_on_1 / f_int_on_1
    ratio2 = f_ext_on_2 / f_int_on_2
    
    return np.maximum(ratio1, ratio2)


def compute_tidal_dominance(
    sub: Dict[str, Any],
    pairs: PairSet,
    sample: Sample,
    h: float,
    box_ckpch: float,
    search_radius_kpc: float = 2000.0
) -> np.ndarray:
    """
    Identify pairs where the primary tidal force 
    on each member is exerted by the other member of the pair.

    Args:
        sub: dict returned by il.groupcat.loadSubhalos(...).
        pairs: PairSet object containing indices (i, j) of candidate pairs.
        sample: Sample object containing subhalo positions and global indices.
        h: Hubble parameter (little h).
        box_ckpch: Periodic box size in ckpc/h.
        search_radius_kpc: Radius to search for potential perturbers (default 2000/2Mpc).

    Returns:
        Boolean mask: True for 'tidally dominant' pairs, False for 'subdominant'.
    """

    m_all = sub["SubhaloMass"].astype(np.float64) * 1e10 / h
    pos_all = (sub["SubhaloPos"].astype(np.float64)) % box_ckpch
    
    # Filter valid subhalos
    valid_mask = (sub["SubhaloFlag"] == 1) & (m_all > 0)
    pos_valid = pos_all[valid_mask]
    m_valid = m_all[valid_mask]
    global_indices_valid = np.where(valid_mask)[0]
    
    # Find neighbors
    tree = cKDTree(pos_valid, boxsize=box_ckpch)
    search_r = search_radius_kpc * h
    neighbors = tree.query_ball_point(sample.pos % box_ckpch, r=search_r)
    
    # Store the global index of the subhalo exerting the max force
    max_perturber_id = np.full(len(sample.pos), -1, dtype=int)
    
    softening_sq = 1.0 # Prevent 1/r^2 divergence
    
    for s_idx in range(len(sample.pos)):
        if not neighbors[s_idx]:
            continue
            
        n_idx_in_valid = neighbors[s_idx]
        n_global_ids = global_indices_valid[n_idx_in_valid]
        
        # Exclude the subhalo itself
        mask = n_global_ids != sample.keep_idx[s_idx]
        if not np.any(mask): continue
        
        # Calculate forces: F ~ M / r^2
        dr = pos_valid[n_idx_in_valid][mask] - (sample.pos[s_idx] % box_ckpch)
        dr -= box_ckpch * np.round(dr / box_ckpch)
        dist_sq = np.sum(dr**2, axis=1) + softening_sq
        
        forces = m_valid[n_idx_in_valid][mask] / dist_sq
        
        max_idx = np.argmax(forces)
        max_perturber_id[s_idx] = n_global_ids[mask][max_idx]

    # Final pairs
    id1 = sample.keep_idx[pairs.i]
    id2 = sample.keep_idx[pairs.j]

    # Tidal champions (source of maximum tidal force)
    champ1 = max_perturber_id[pairs.i]
    champ2 = max_perturber_id[pairs.j]
    
    is_dominant = (champ1 == id2) & (champ2 == id1)
    
    return is_dominant