import illustris_python as il
import numpy as np
from scipy.spatial import cKDTree


def load_halos_subhalos_and_header_from_snapshot(
        basePath, 
        snapshot, 
        halo_fields=["GroupFirstSub", "GroupNsubs"], 
        subhalo_fields=[
            "SubhaloPos",       # Spatial position within the periodic box (of the particle with the minium gravitational potential energy). Comoving coordinate. Units: ckpc/h 
            "SubhaloVel",       # Peculiar velocity of the group, computed as the sum of the mass weighted velocities of all particles/cells in this group, of all types. No unit conversion is needed. Units: km/s
            "SubhaloMass",      # Total mass of all member particle/cells which are bound to this Subhalo, of all types. Particle/cells bound to subhaloes of this Subhalo are NOT accounted for. Units: 1e10 Msun/h
            "SubhaloMassType",  # Total mass of all member particle/cells which are bound to this Subhalo, separated by type. Particle/cells bound to subhaloes of this Subhalo are NOT accounted for. Units: 1e10 Msun/h
            "SubhaloSFRinRad",  # Sum of the individual star formation rates of all gas cells in this subhalo, but restricted to cells within the radius of 𝑉𝑚𝑎𝑥. Units Msun/yr
            "SubhaloFlag",      # Flag field indicating suitability of this subhalo for certain types of analysis. If zero, this subhalo should generally be excluded, and is not thought to be of cosmological origin.
            "SubhaloGrNr"       # Index into the Group table of the FOF host/parent of this Subhalo. 
        ]
    ):
    halos = il.groupcat.loadHalos(basePath, snapshot, fields=halo_fields)
    subhalos = il.groupcat.loadSubhalos(basePath, snapshot, fields=subhalo_fields)
    header = il.groupcat.loadHeader(basePath, snapshot)
    return halos, subhalos, header


def filter_cosmological_central_subhalos(halos, subhalos):
    first = halos["GroupFirstSub"]

    # We create an array of length sub["count"] and mark those indices listed in GroupFirstSub (excluding -1). 
    # GroupFirstSub points into the Subhalo table, so this is the direct way to label central subhalos.
    is_central = np.zeros(subhalos["count"], dtype=bool)

    valid = first[first >= 0]     # drop halos with no subhalos
    is_central[valid] = True

    # Combine centrality with the “useful galaxy” flag
    # SubhaloFlag == 1 indicates that the subhalo is of cosmological origin and thus suitable for analysis
    keep_filter = (subhalos["SubhaloFlag"] == 1) & is_central

    filtered_subhalos = {
    'pos_c':    subhalos["SubhaloPos"][keep_filter],
    'vel_c':    subhalos["SubhaloVel"][keep_filter],
    'mtype_c':  subhalos["SubhaloMassType"][keep_filter],
    'grnr_c':   subhalos["SubhaloGrNr"][keep_filter]
    }

    print("All subhalos:", subhalos["count"])
    print("Centrals (raw):", is_central.sum())
    print("Centrals with SubhaloFlag==1:", keep_filter.sum())

    # Optional: check how many halos have no subhalos
    print("Halos with no subhalos:", np.sum(first < 0))

    return filtered_subhalos


def filter_with_stellar_mass(subhalos, mstar_min=1e10, mstar_max=1e12, h=0.6774):
    # Stellar mass from SubhaloMassType:
    # SubhaloMassType is in 1e10 Msun/h, and index 4 corresponds to stars & wind particles.
    mstar = subhalos['mtype_c'][:, 4] * 1e10 / h  # Msun

    # Optional: dark matter mass (same units logic, index 1 is DM).
    mdm = subhalos['mtype_c'][:, 1] * 1e10 / h  # Msun

    # Choose an initial stellar-mass window.
    # This is not a "correct" MW/M31 definition, just a practical starting point to reduce the sample size.
    mstar_min = 1e10
    mstar_max = 1e12

    # Build the filter.
    stellar_mass_filter = (mstar >= mstar_min) & (mstar <= mstar_max)

    # Apply it to the central sample arrays.
    filtered_subhalos = {
    'pos_s':    subhalos['pos_c'][stellar_mass_filter],
    'vel_s':    subhalos['vel_c'][stellar_mass_filter],
    'mstar_s':  mstar[stellar_mass_filter],
    'mdm_s':    mdm[stellar_mass_filter],
    'grnr_s':   subhalos['grnr_c'][stellar_mass_filter],
    }

    print("\n\nCentrals before stellar-mass filter:", len(subhalos['pos_c']))
    print("Centrals after stellar-mass filter:", len(filtered_subhalos['pos_s']))
    print("h used:", h)
    print("Stellar mass range [Msun]:", mstar_min, "to", mstar_max)

    return filtered_subhalos


def pair_finding(subhalos, header, r_min=500.0, r_max=1000.0, h=0.6774):
    box = header["BoxSize"]  # ckpc/h
    print("\n\nBoxSize [ckpc/h]:", box)

    # Distances:
    # SubhaloPos is in ckpc/h. The target window is 500–1000 kpc (physical).
    # At z=0, converting kpc -> ckpc/h is: ckpc/h = kpc * h.
    r_min = r_min * h
    r_max = r_max * h

    # Periodic box size in ckpc/h from the header.
    box = header["BoxSize"]  # ckpc/h 

    # Build a periodic k-d tree.
    # SciPy cKDTree supports periodic boundaries via the 'boxsize' argument. :contentReference[oaicite:3]{index=3}
    tree = cKDTree(subhalos['pos_s'], boxsize=box)

    # Get all unique index pairs (i < j) with separation <= r_max (in ckpc/h).
    pairs = tree.query_pairs(r_max, output_type='ndarray')
    print("Candidate pairs within r_max (before extra cuts):", pairs.shape[0])

    # Split indices.
    i = pairs[:, 0]
    j = pairs[:, 1]

    # Exclude pairs that live in the same FoF halo.
    # SubhaloGrNr maps each subhalo to its parent FoF halo index. :contentReference[oaicite:4]{index=4}
    diff_host = subhalos['grnr_s'][i] != subhalos['grnr_s'][j]
    i = i[diff_host]
    j = j[diff_host]

    print("Pairs after excluding same-host pairs:", i.shape[0])

    # Compute separations with the minimum-image convention.
    # This is needed to get the correct displacement vector in a periodic box.
    dr = subhalos['pos_s'][j] - subhalos['pos_s'][i]
    dr -= box * np.round(dr / box)

    dist = np.linalg.norm(dr, axis=1)

    # Apply the lower bound r_min.
    in_window = dist >= r_min
    i = i[in_window]
    j = j[in_window]
    dr = dr[in_window]
    dist = dist[in_window]

    print("Pairs in [r_min, r_max]:", i.shape[0])

    # Convert separation to physical kpc for reporting and plotting.
    dist_kpc = dist / h

    # Relative velocity vector (km/s).
    # SubhaloVel is peculiar velocity in km/s. :contentReference[oaicite:5]{index=5}
    dv = subhalos['vel_s'][j] - subhalos['vel_s'][i]

    # Radial and tangential relative velocities.
    rhat = dr / dist[:, None]
    v_r = np.einsum("ij,ij->i", dv, rhat)
    v2 = np.einsum("ij,ij->i", dv, dv)
    v_t = np.sqrt(np.maximum(0.0, v2 - v_r * v_r))

    result_set = {
        'i': i,
        'j': j,
        'dr': dr,
        'dist': dist,
        'dist_kpc': dist_kpc,
        'rhat': rhat,
        'v_r': v_r,
        'v2': v2,
        'v_t': v_t
    }

    print("Separation stats [kpc]: min/median/max =", np.min(dist_kpc), np.median(dist_kpc), np.max(dist_kpc))
    print("v_r stats [km/s]: min/median/max =", np.min(v_r), np.median(v_r), np.max(v_r))
    print("v_t stats [km/s]: min/median/max =", np.min(v_t), np.median(v_t), np.max(v_t))

    return result_set


def find_local_group_analogues(path, snapshot, mstar_min=1e10, mstar_max=1e12, r_min=500.0, r_max=1000.0, h=0.6774):
    halos, subhalos, header = load_halos_subhalos_and_header_from_snapshot(path, snapshot)
    subhalos_c = filter_cosmological_central_subhalos(halos, subhalos)
    subhalos_sm = filter_with_stellar_mass(subhalos_c, mstar_min, mstar_max, h)
    return pair_finding(subhalos_sm, header, r_min, r_max, h)


# Example usage with Working directory being in the root of repository
#import os
#find_local_group_analogues(os.path.abspath("./tng300/outputs/"), 99)
