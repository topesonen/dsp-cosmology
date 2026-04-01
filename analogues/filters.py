from __future__ import annotations
from typing import Dict, Any
import numpy as np
from scipy.spatial import cKDTree
from .data_structures import PairSet, Sample, SelectionConfig, FilterConfig


def filter_by_isolation_no_third_factor(
    sub: Dict[str, Any],
    pairs: PairSet,
    sample: Sample,
    selection_config: SelectionConfig,
    filter_config: FilterConfig
) -> np.ndarray:
    """Isolation filter: reject pairs with a very massive third object in the same FoF group(s).

    For each pair (a,b), reject if there exists a third subhalo in the relevant FoF group(s)
    whose stellar mass >= x * max(Mstar[a], Mstar[b]).

    The third-object search is done on all subhalos with SubhaloFlag==1 and Mstar>0
    but restricted to the FoF groups that appear in at least one pair member.

    Args:
        sub (Dict[str, Any]): Loaded subhalos.
        pairs (PairSet): Subhalo pairs within given separation.
        sample (Sample): Subhalos with masses in given range.
        selection_config (SelectionConfig): Selection criteria and simulation constants.
        filter_config (SelectionConfig): Filtering criteria.

    Returns:
        Boolean mask, True for pairs that pass isolation.
    """
    pair_i = pairs.i
    pair_j = pairs.j

    sample_grnr = sample.grnr
    sample_keep_idx_global = sample.keep_idx

    h = selection_config.hubble_param
    x = filter_config.third_massive_factor

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


def filter_by_isolation_no_intruder(
    sub: Dict[str, Any],
    pairs: PairSet,
    sample: Sample,
    selection_config: SelectionConfig,
    filter_config: FilterConfig
) -> np.ndarray:
    """
    Isolation filter: reject pairs if any third subhalo within r_iso_kpc 
    of the pair's Center of Mass is more massive than the pair's combined mass.

    Args:
        sub (Dict[str, Any]): Loaded subhalos.
        pairs (PairSet): Subhalo pairs within given separation.
        sample (Sample): Subhalos with masses in given range.
        selection_config (SelectionConfig): Selection criteria and simulation constants.
        filter_config (SelectionConfig): Filtering criteria.

    Returns:
        Boolean mask, True for pairs that pass isolation.
    """

    pair_i = pairs.i
    pair_j = pairs.j

    sample_pos = sample.pos
    sample_keep_idx_global = sample.keep_idx

    h = selection_config.hubble_param
    box_ckpch = selection_config.box_side_length
    r_iso_kpc = filter_config.density_radius
    mass_factor = filter_config.intruder_factor

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


def filter_by_force_ratio(
    sub: Dict[str, Any], 
    pairs: PairSet,
    sample: Sample,
    selection_config: SelectionConfig,
    filter_config: FilterConfig
) -> np.ndarray:

    """ Filter subhalo pairs to a given force ratio range.

    Args:
        sub (Dict[str, Any]): Loaded subhalos.
        pairs (PairSet): Subhalo pairs within given separation.
        sample (Sample): Subhalos with masses in given range.
        selection_config (SelectionConfig): Selection criteria and simulation constants.
        filter_config (SelectionConfig): Filtering criteria.

    Returns:
        Boolean mask, True for pairs that pass force ratio condition.
    """

    return (
        (pairs.force_ratio >= filter_config.force_ratio_min)
        & (pairs.force_ratio <= filter_config.force_ratio_max)
    )


def filter_by_tidal_dominance(
    sub: Dict[str, Any], 
    pairs: PairSet,
    sample: Sample,
    selection_config: SelectionConfig,
    filter_config: FilterConfig
) -> np.ndarray:
    """ Filter subhalo pairs to be either 
    tidally dominant or tidally sub-dominant.

    Args:
        sub (Dict[str, Any]): Loaded subhalos.
        pairs (PairSet): Subhalo pairs within given separation.
        sample (Sample): Subhalos with masses in given range.
        selection_config (SelectionConfig): Selection criteria and simulation constants.
        filter_config (SelectionConfig): Filtering criteria.

    Returns:
        Boolean mask, True for pairs that pass tidal dominance condition.
    """

    if filter_config.tidally_dominant:
        return pairs.is_tidally_dominant
    elif not filter_config.tidally_dominant:
        return ~pairs.is_tidally_dominant
    else:
        return np.ones(len(pairs.i), dtype=bool)


def filter_by_vel(
    sub: Dict[str, Any], 
    pairs: PairSet,
    sample: Sample,
    selection_config: SelectionConfig,
    filter_config: FilterConfig
) -> np.ndarray:
    """ Filter subhalo pairs to a given relative velocity range in terms of
    radial and transverse velocity components, and total relative velocity.

    Args:
        sub (Dict[str, Any]): Loaded subhalos.
        pairs (PairSet): Subhalo pairs within given separation.
        sample (Sample): Subhalos with masses in given range.
        selection_config (SelectionConfig): Selection criteria and simulation constants.
        filter_config (SelectionConfig): Filtering criteria.

    Returns:
        Boolean mask, True for pairs that pass velocity conditions.
    """

    vt = pairs.vt
    vr = pairs.vr
    v_tot = np.sqrt(vr**2 + vt**2)

    mask = np.ones(len(vr), dtype=bool)

    if filter_config.vt_min is not None:
        mask &= (vt >= filter_config.vt_min)
    
    if filter_config.vt_max is not None:
        mask &= (vt <= filter_config.vt_max)
        
    if filter_config.vr_min is not None:
        mask &= (vr >= filter_config.vr_min)
        
    if filter_config.vr_max is not None:
        mask &= (vr <= filter_config.vr_max)

    if filter_config.v_tot_min is not None:
        mask &= (v_tot >= filter_config.v_tot_min)
        
    if filter_config.v_tot_max is not None:
        mask &= (v_tot <= filter_config.v_tot_max)

    return mask




    