"""Local Group analogue utilities for IllustrisTNG group catalogs.

This module contains:
- Selection of central + largest satellite (by stellar mass) per FoF group
- Pair finding within a distance window in a periodic box

Assumptions:
- TNG group catalogs loaded via illustris_python (il.groupcat.loadSubhalos/loadHalos/loadHeader)
- Snapshot is typically z=0 (snap=99 for TNG300-1), but the utilities are general
"""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
from scipy.spatial import cKDTree
from tabulate import tabulate
import illustris_python as il
import h5py
from .data_structures import PairSet, Sample, SelectionConfig


class AnaloguePipeline:
    def __init__(self, pairs: PairSet):
        self.pairs = pairs
        self.history = {"initial": len(pairs.i)}
        self.labels: list[str] = []
        self.masks: list[np.ndarray] = []
        self._cutflow: list[dict[str, float | int | str]] = [{
            "label": "initial",
            "before": len(pairs.i),
            "after": len(pairs.i),
            "frac_kept": 1.0,
        }]

    def apply_filter(self, filter_name: str, mask: np.ndarray) -> None:
        # Applies a boolean mask and logs how many pairs survived.
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("mask must be a 1D boolean array")
        n_before = len(self.pairs.i)
        if mask.shape[0] != n_before:
            raise ValueError(f"mask length {mask.shape[0]} does not match number of pairs {n_before}")

        self.labels.append(filter_name)
        self.masks.append(mask.copy())

        self.pairs = self.pairs.apply_mask(mask)
        n_after = len(self.pairs.i)
        self.history[filter_name] = n_after
        frac = (n_after / n_before) if n_before > 0 else 0.0
        self._cutflow.append({
            "label": filter_name,
            "before": n_before,
            "after": n_after,
            "frac_kept": frac,
        })

    def get_cutflow(self) -> list[dict[str, float | int | str]]:
        """Return per-step cutflow entries with before/after counts and kept fraction."""
        return list(self._cutflow)
        
    def print_cutflow(self) -> None:
        """Prints a professional cutflow table with survival statistics."""
        if not self._cutflow:
            print("Cutflow is empty.")
            return

        initial_count = self._cutflow[0]["before"]
        table_data = []

        for i, row in enumerate(self._cutflow):
            name = row["label"]
            after = row["after"]
            
            # Cumulative: % remaining from the very beginning
            cum_pct = (after / initial_count * 100) if initial_count > 0 else 0.0
            
            # Step-wise: % remaining from the previous step
            if i == 0:
                step_pct = 100.0
            else:
                before = row["before"]
                step_pct = (after / before * 100) if before > 0 else 0.0

            table_data.append([
                name, 
                f"{after:,}", 
                f"{step_pct:>6.1f}%", 
                f"{cum_pct:>6.1f}%"
            ])

        print("\n[PIPELINE] Sample Reduction Cutflow")
        headers = ["Filter Step", "Pairs Remaining", "Step Yield", "Total Survival"]
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", stralign="left", numalign="right"))


def load_header(basePath: str, snap: int, verbose: bool = True) -> Dict[str, float]:
    """Load commonly used header constants.

    Returns:
        dict with keys:
        - hubble_param: HubbleParam
        - box_side_length: BoxSize (ckpc/h)
    """
    header = il.groupcat.loadHeader(basePath, snap)
    hubble_param = float(header["HubbleParam"])
    box_side_length = float(header["BoxSize"]) / hubble_param

    if verbose:
        header_data = [
            ["Parameter", "Value", "Units"],
            ["Hubble (h)", hubble_param, "dimensionless"],
            ["Box Side", f"{box_side_length:.2f}", "kpc"]
        ]

        print("\n[DIAGNOSTIC] Simulation Header Constants")
        print(tabulate(header_data, tablefmt="fancy_grid"))

    return {
        "hubble_param": hubble_param,
        "box_side_length": box_side_length,
    }

def load_catalog_with_sample_ids(
    path: str, 
    snapshot: str, 
    sample_ids: np.ndarray, 
    fields: List[str]
) -> Dict[str, np.ndarray]:
    """Extract catalog fields for a subset of sample IDs.
    
    Reads multiple datasets from an HDF5 catalog file and matches them to sample IDs
    using a dictionary lookup. Fields are only returned for IDs present in the catalog;
    missing IDs are filled with np.nan.
    
    Args:
        path: Absolute path to the HDF5 catalog file (e.g., 'stellar_circs.hdf5').
        snapshot: Group name within the HDF5 file (e.g., 'Snapshot_99').
        sample_ids: Array of subhalo IDs to extract values for (length M_sample).
        fields: List of dataset names to load from the catalog snapshot.
    
    Returns:
        Dictionary mapping field names to numpy arrays of length len(sample_ids).
        Each array is aligned to sample_ids order; missing values are np.nan.
    
    Example:
        >>> result = get_catalog_fields_with_sample_ids(
        ...     path='../tng300/catalog_c/stellar_circs.hdf5',
        ...     snapshot='Snapshot_99',
        ...     sample_ids=sample.keep_idx,
        ...     fields=['CircAbove07Frac', 'SpecificAngMom']
        ... )
        >>> circ_values = result['CircAbove07Frac']  # length == len(sample.keep_idx)
    """
    catalog_dict: Dict[str, np.ndarray] = {}
    
    with h5py.File(path, "r") as f:
        snap = f[snapshot]
        
        # Load the catalog's subhalo ID array (index into catalog)
        catalog_ids = snap["SubfindID"][...].astype(int)   # length N_catalog
        
        for field in fields:
            # Load the field values from catalog (same length as catalog_ids)
            catalog_vals = snap[field][...]  # length N_catalog
            
            # Build ID → value mapping for fast lookup
            id_to_val = dict(zip(catalog_ids, catalog_vals))
            
            # Extract values for sample_ids; use np.nan for missing IDs
            catalog_dict[field] = np.array(
                [id_to_val.get(sid, np.nan) for sid in sample_ids],
                dtype=catalog_vals.dtype
            )
    
    return catalog_dict

def generate_sample(
    sub: Dict[str, Any],
    basePath: str,
    snap: int,
    selection_config: SelectionConfig,
    catalog_c_path: str,
    verbose: bool = True,
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
        selection_config: object containing the selection criteria 
            and the simulation constants
        catalog_c_path: catalog_c path to stellar_circs.hdf5 file
        verbose: If True, print diagnostics

    Returns:
        Sample containing global indices and sliced arrays.
    """
    h = selection_config.hubble_param
    mstar_min = selection_config.m_stellar_min
    mstar_max = selection_config.m_stellar_max

    halo_cat = il.groupcat.loadHalos(basePath, snap, fields=[
        "GroupFirstSub", "GroupNsubs", "Group_M_Crit200", "Group_R_Crit200", 
    ])
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
    
    G = 4.30091e-6 
    selected_grnr = grnr_all[keep_idx]
    r200c=halo_cat["Group_R_Crit200"][selected_grnr].astype(np.float64) / h
    m200c=halo_cat["Group_M_Crit200"][selected_grnr].astype(np.float64) / h

    v200c = np.sqrt(G * m200c / np.where(r200c > 0, r200c, 1.0))
    v200c[r200c <= 0] = 0

    photometrics = sub["SubhaloStellarPhotometrics"][keep_idx][:, [0, 3, 4, 5]].astype(np.float64)
    u_band, k_band, g_band, r_band = photometrics[:, 0], photometrics[:, 1], photometrics[:, 2], photometrics[:, 3]
    is_blue = g_band - r_band < selection_config.blue_threshold_gr
    is_red = ~is_blue

    if verbose:
        _print_sample_diagnostics(
            sub=sub,
            good=good,
            in_window=in_window,
            central_keep=central_keep,
            sat_mask=sat_mask,
            keep_idx=keep_idx
        )

    catalog_c = load_catalog_with_sample_ids(
        path=catalog_c_path, 
        snapshot='Snapshot_' + str(snap),
        sample_ids=keep_idx,
        fields = [
                "CircAbove07Frac",
                "CircAbove07Frac_allstars",
                "CircAbove07MinusBelowNeg07Frac",
                "CircAbove07MinusBelowNeg07Frac_allstars",
                "CircTwiceBelow0Frac",
                "CircTwiceBelow0Frac_allstars",
                "MassTensorEigenVals",
                "ReducedMassTensorEigenVals",
                "SpecificAngMom",
                "SpecificAngMom_allstars",
            ]
        )
    
    is_disc = catalog_c['CircAbove07Frac_allstars'] > selection_config.disc_threshold

    return Sample(
        keep_idx=keep_idx,
        grnr=selected_grnr,
        is_central=is_central[keep_idx],
        is_blue=is_blue,
        is_red=is_red,
        pos=sub["SubhaloPos"][keep_idx],
        vel=sub["SubhaloVel"][keep_idx],
        m_gas=mtype[:, 0].astype(np.float64) * 1e10 / h,
        m_dark_matter=mtype[:, 1].astype(np.float64) * 1e10 / h,
        m_tracers=mtype[:, 3].astype(np.float64) * 1e10 / h,
        m_stellar=mtype[:, 4].astype(np.float64) * 1e10 / h,
        m_black_hole=mtype[:, 5].astype(np.float64) * 1e10 / h,
        m_tot = sub["SubhaloMass"][keep_idx].astype(np.float64) * 1e10 / h,
        r200c=r200c,
        v200c=v200c,
        m200c=m200c,
        u_band=u_band,
        g_band=g_band,
        r_band=r_band,
        k_band=k_band,
        circ=catalog_c['CircAbove07Frac_allstars'],
        ang_mom=catalog_c['SpecificAngMom_allstars'],
        eigen=catalog_c['MassTensorEigenVals'],
        is_disc=is_disc
    )


def _print_sample_diagnostics(sub, good, in_window, central_keep, sat_mask, keep_idx):
    """Prints a breakdown of the selection pipeline results."""
    
    total_subhalos = len(sub["SubhaloFlag"])
    n_good = np.sum(good)
    n_in_window = np.sum(in_window & good)
    n_centrals = np.sum(central_keep)
    n_sats_eligible = np.sum(sat_mask)
    n_final = len(keep_idx)
    
    # Calculate the 'loss' at each step
    stats = [
        ["Total Subhalos in Catalog", total_subhalos, "100.0%"],
        ["Passed Quality Flags (SubhaloFlag==1)", n_good, f"{(n_good/total_subhalos)*100:.1f}%"],
        ["Inside Stellar Mass Window", n_in_window, f"{(n_in_window/n_good)*100:.1f}% of good"],
        ["Eligible Centrals (Selected Groups)", n_centrals, f"{(n_centrals/n_in_window)*100:.1f}% of window"],
        ["Eligible Satellites in those Groups", n_sats_eligible, f"{(n_sats_eligible/n_in_window)*100:.1f}% of window"],
        ["Final Sample (Centrals + Max Satellite)", n_final, f"{(n_final/total_subhalos)*100:.2f}% of total"]
    ]

    print("\n[DIAGNOSTIC] Sample Selection")
    print(tabulate(stats, headers=["Selection Step", "Count", "Yield"], tablefmt="fancy_grid"))

    n_groups = len(np.unique(sub["SubhaloGrNr"][keep_idx]))
    solo_centrals = n_centrals - (n_final - n_centrals)
    
    group_stats = [
        ["Total Unique Groups", n_groups],
        ["Groups with Satellites", n_final - n_centrals],
        ["Groups with Central Only", solo_centrals]
    ]

    print(tabulate(group_stats, tablefmt="simple"))
    print("")


def find_pairs(
    sub: Dict[str, Any],
    sample: Sample,
    selection_config: SelectionConfig,
    verbose: bool = True
) -> PairSet:
    """Find pairs within [r_min_kpc, r_max_kpc] using a periodic cKDTree and
    compute pairwise kinematics for the found pairs.

    Args:
        sub: dict returned by il.groupcat.loadSubhalos(...).
        sample: mass filtered sample set
        selection_config: object containing the selection criteria 
            and the simulation constants
        verbose: If True, print diagnostics

    Returns:
        PairSet with indices into the input arrays and derived pairwise kinematics.
    """
    pos = sample.pos
    vel = sample.vel
    keep_idx_global = sample.keep_idx
    grnr = sample.grnr

    h = selection_config.hubble_param
    box_ckpch = selection_config.box_side_length * h
    r_min = selection_config.r_min
    r_max = selection_config.r_max

    r_min = r_min * h
    r_max = r_max * h

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

    separation = dist / h

    dv = vel[j] - vel[i]
    rhat = dr / dist[:, None]
    vr = np.einsum("ij,ij->i", dv, rhat)

    v2 = np.einsum("ij,ij->i", dv, dv)
    vt = np.sqrt(np.maximum(0.0, v2 - vr * vr))

    have_same_host = (grnr[i] == grnr[j])

    total_virial_mass = np.zeros(len(i))
    total_virial_mass[have_same_host] = sample.m200c[i][have_same_host]
    total_virial_mass[~have_same_host] = sample.m200c[i][~have_same_host] + sample.m200c[j][~have_same_host]
    log_tot_virial_mass = np.log10(total_virial_mass * 1e10 / h)

    m_i = sample.m_tot[i]
    m_j = sample.m_tot[j]
    log_mass_ratio=np.log10(np.minimum(m_i, m_j) / np.maximum(m_i, m_j))

    force_ratio = _compute_force_ratios(sub, i, j, separation, pos, keep_idx_global, h, box_ckpch)
    is_tidally_dominant = _determine_tidal_dominance(sub, i, j, pos, keep_idx_global, h, box_ckpch)

    is_blue_blue = sample.is_blue[i] & sample.is_blue[j]
    is_red_red = sample.is_red[i] & sample.is_red[j]
    is_blue_red = (sample.is_red[i] & sample.is_blue[j]) | (sample.is_blue[i] & sample.is_red[j])

    if verbose:
        _print_pair_diagnostics(
            i, have_same_host, force_ratio, is_tidally_dominant,
            is_blue_blue, is_red_red, is_blue_red, separation, vr, vt, log_mass_ratio
        )

    return PairSet(
        i=i,
        j=j,
        have_same_host=have_same_host,
        is_tidally_dominant=is_tidally_dominant,
        separation=separation,
        is_blue_blue=is_blue_blue,
        is_red_red=is_red_red,
        is_blue_red=is_blue_red,
        vr=vr,
        vt=vt,
        force_ratio=force_ratio,
        log_tot_virial_mass=log_tot_virial_mass,
        log_mass_ratio=log_mass_ratio
    )


def _print_pair_diagnostics(i, have_same_host, force_ratio, is_tidally_dominant, 
                           is_blue_blue, is_red_red, is_blue_red, 
                           separation, vr, vt, log_mass_ratio):
    """Prints detailed statistics for the found subhalo pairs."""
    
    n_pairs = i.size
    if n_pairs == 0:
        print("\n[DIAGNOSTIC] No pairs found.")
        return

    mean_sep = np.mean(separation)
    mean_v_rel = np.mean(np.sqrt(vr**2 + vt**2))
    mean_mu = np.mean(10**log_mass_ratio)

    data = [
        ["Total Pairs Found", n_pairs, "100.0%"],
        ["Same-host (FoF)", np.sum(have_same_host), f"{np.mean(have_same_host)*100:.1f}%"],
        ["Different-host", np.sum(~have_same_host), f"{np.mean(~have_same_host)*100:.1f}%"],
        ["Tidally Dominant", np.sum(is_tidally_dominant), f"{np.mean(is_tidally_dominant)*100:.1f}%"],
        ["Force Ratio < 0.5", np.sum(force_ratio < 0.5), f"{np.mean(force_ratio < 0.5)*100:.1f}%"],
        ["Blue-Blue", np.sum(is_blue_blue), f"{np.mean(is_blue_blue)*100:.1f}%"],
        ["Red-Red", np.sum(is_red_red), f"{np.mean(is_red_red)*100:.1f}%"],
        ["Mixed Blue-Red", np.sum(is_blue_red), f"{np.mean(is_blue_red)*100:.1f}%"],
    ]

    print("\n[DIAGNOSTIC] Pair Finding Results")
    print(tabulate(data, headers=["Category", "Count", "Share"], tablefmt="fancy_grid"))
    
    phys_data = [
        ["Mean Separation", f"{mean_sep:.2f} kpc"],
        ["Mean Relative Velocity", f"{mean_v_rel:.2f} km/s"],
        ["Mean Mass Ratio (μ)", f"{mean_mu:.3f}"]
    ]
    print(tabulate(phys_data, tablefmt="simple"))
    print("")


def _compute_force_ratios(
    sub: Dict[str, Any],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_dist_kpc: np.ndarray,
    sample_pos: np.ndarray,
    sample_keep_idx_global: np.ndarray,
    h: float,
    box_ckpch: float,
    search_radius_kpc: float = 2000.0,
) -> np.ndarray:
    """
    Compute the maximum ratio of external tidal force 
    to internal gravitational force for each galaxy pair.

    Args:
        sub: dict returned by il.groupcat.loadSubhalos(...).
        pair_i, pair_j: Pair indices into the sample arrays.
        pair_dist_kpc: Separation between each pair.
        sample_pos: Positions (ckpc/h) of the subhalos in the sample.
        sample_keep_idx_global: global Subhalo table indices per sample object.
        h: Hubble parameter (little h).
        box_ckpch: Periodic box size in ckpc/h.
        search_radius_kpc: Radius to search for external perturbers (default 2000/2Mpc).

    Returns:
        np.ndarray: Force ratio F for each pair.
    """
   
    # Find local neighbors
    neighbors, pos_valid, m_all, m_valid, global_ids_valid = _get_local_neighbors(
        sample_pos, sub, h, box_ckpch, search_radius_kpc
    )
    
    # Initialize arrays
    max_f1 = np.zeros(len(sample_pos))    # Strongest force
    id_f1 = np.full(len(sample_pos), -1)  # ID of strongest
    max_f2 = np.zeros(len(sample_pos))    # Second strongest force

    softening_sq = 1.0 # Prevent 1/r^2 divergence
    
    # Find the disturbers for each subhalo
    for s_idx in range(len(sample_pos)):

        # Skip if there are no neighbors
        if not neighbors[s_idx]:
            continue
        
        # Find the global index for each neighbor
        n_idx_in_valid = neighbors[s_idx]
        n_global_ids = global_ids_valid[n_idx_in_valid]
        
        # Exclude the subhalo itself
        self_mask = n_global_ids != sample_keep_idx_global[s_idx]
        if not np.any(self_mask): continue

        # Compute all separations
        dr = pos_valid[n_idx_in_valid][self_mask] - (sample_pos[s_idx] % box_ckpch)
        dr -= box_ckpch * np.round(dr / box_ckpch)
        dist_sq = np.sum(dr**2, axis=1) + softening_sq
        
        # Compute all gravitational attractions
        all_forces = m_valid[n_idx_in_valid][self_mask] / dist_sq
        
        # Sort forces to get the top two
        if len(all_forces) >= 2:
            sort_idx = np.argsort(all_forces)[-2:]
            max_f1[s_idx] = all_forces[sort_idx[1]]
            id_f1[s_idx] = n_global_ids[self_mask][sort_idx[1]]
            max_f2[s_idx] = all_forces[sort_idx[0]]
        elif len(all_forces) == 1:
            max_f1[s_idx] = all_forces[0]
            id_f1[s_idx] = n_global_ids[self_mask][0]

    # LG analogue pair
    idx1, idx2 = pair_i, pair_j
    global_id1 = sample_keep_idx_global[idx1]
    global_id2 = sample_keep_idx_global[idx2]
    
    # Internal force between the pair
    r_pair_sq = (pair_dist_kpc * h)**2 + softening_sq
    f_int_on_1 = m_all[global_id2] / r_pair_sq
    f_int_on_2 = m_all[global_id1] / r_pair_sq
    
    # Select max external force,
    f_ext_on_1 = np.where(id_f1[idx1] == global_id2, max_f2[idx1], max_f1[idx1])
    f_ext_on_2 = np.where(id_f1[idx2] == global_id1, max_f2[idx2], max_f1[idx2])
    
    # Compute final ratios
    ratio1 = f_ext_on_1 / f_int_on_1
    ratio2 = f_ext_on_2 / f_int_on_2
    
    return np.maximum(ratio1, ratio2)


def _determine_tidal_dominance(
    sub: Dict[str, Any],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    sample_pos: np.ndarray,
    sample_keep_idx_global: np.ndarray,
    h: float,
    box_ckpch: float,
    search_radius_kpc: float = 2000.0
) -> np.ndarray:
    """
    Identify pairs where the primary tidal force 
    on each member is exerted by the other member of the pair.

    Args:
        sub: dict returned by il.groupcat.loadSubhalos(...).
        pair_i, pair_j: Pair indices into the sample arrays.
        pair_dist_kpc: Separation between each pair.
        sample_pos: Positions (ckpc/h) of the subhalos in the sample.
        h: Hubble parameter (little h).
        box_ckpch: Periodic box size in ckpc/h.
        search_radius_kpc: Radius to search for potential perturbers (default 2000/2Mpc).

    Returns:
        Boolean mask: True for 'tidally dominant' pairs, False for 'subdominant'.
    """

    # Find local neighbors
    neighbors, pos_valid, m_all, m_valid, global_ids_valid = _get_local_neighbors(
        sample_pos, sub, h, box_ckpch, search_radius_kpc
    )
    
    # Store the global index of the subhalo exerting the max force
    max_perturber_id = np.full(len(sample_pos), -1, dtype=int)
    
    softening_sq = 1.0 # Prevent 1/r^2 divergence
    
    # Find the disturbers for each subhalo
    for s_idx in range(len(sample_pos)):
        
        # Skip if there are no neighbors
        if not neighbors[s_idx]:
            continue
        
        # Find the global index for each neighbor
        n_idx_in_valid = neighbors[s_idx]
        n_global_ids = global_ids_valid[n_idx_in_valid]
        
        # Exclude the subhalo itself
        mask = n_global_ids != sample_keep_idx_global[s_idx]
        if not np.any(mask): continue
        
        # Compute all separations
        dr = pos_valid[n_idx_in_valid][mask] - (sample_pos[s_idx] % box_ckpch)
        dr -= box_ckpch * np.round(dr / box_ckpch)
        dist_sq = np.sum(dr**2, axis=1) + softening_sq
        
        # Compute all gravitational attractions
        forces = m_valid[n_idx_in_valid][mask] / dist_sq
        
        # Find the subhalo exerting the maximum attraction
        max_idx = np.argmax(forces)
        max_perturber_id[s_idx] = n_global_ids[mask][max_idx]

    # LG analogue pair
    idx1 = sample_keep_idx_global[pair_i]
    idx2 = sample_keep_idx_global[pair_j]

    # Tidal champions for each pair (source of maximum tidal force)
    champ1 = max_perturber_id[pair_i]
    champ2 = max_perturber_id[pair_j]
    
    # The pair is tidally dominant if they are mutual champions
    is_dominant = (champ1 == idx2) & (champ2 == idx1)
    
    return is_dominant

 
def _get_local_neighbors(sample_pos, sub, h, box_ckpch, search_radius_kpc=2000.0):
    """
    Helper to identify valid neighbors within a physical search radius.
    """

    # Initialize masses and positions
    m_all = sub["SubhaloMass"].astype(np.float64) * 1e10 / h
    pos_all = (sub["SubhaloPos"].astype(np.float64)) % box_ckpch
    
    # Filter to include only valid subhalos
    valid_mask = (sub["SubhaloFlag"] == 1) & (m_all > 0)
    pos_valid = pos_all[valid_mask]
    m_valid = m_all[valid_mask]
    global_indices_valid = np.where(valid_mask)[0]
    
    # Find all neighbors within `search_radius_kpc`
    tree = cKDTree(pos_valid, boxsize=box_ckpch)
    search_r = search_radius_kpc * h
    neighbors = tree.query_ball_point(sample_pos % box_ckpch, r=search_r)
    
    return neighbors, pos_valid, m_all, m_valid, global_indices_valid