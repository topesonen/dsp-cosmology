import illustris_python as il
import numpy as np
from scipy.spatial import cKDTree
from typing import Any


def load_halos_subhalos_and_header_from_snapshot(
    basePath: str,
    snapshot: int,
    halo_fields: list[str] = ["GroupFirstSub", "GroupNsubs"],
    subhalo_fields: list[str] = [
        "SubhaloPos",  # Spatial position within the periodic box (of the particle with the minimum gravitational potential energy). Comoving coordinate. Units: ckpc/h
        "SubhaloVel",  # Peculiar velocity of the group, computed as the sum of the mass weighted velocities of all particles/cells in this group, of all types. No unit conversion is needed. Units: km/s
        "SubhaloMass",  # Total mass of all member particle/cells which are bound to this Subhalo, of all types. Particle/cells bound to subhaloes of this Subhalo are NOT accounted for. Units: 1e10 Msun/h
        "SubhaloMassType",  # Total mass of all member particle/cells which are bound to this Subhalo, separated by type. Particle/cells bound to subhaloes of this Subhalo are NOT accounted for. Units: 1e10 Msun/h
        "SubhaloSFRinRad",  # Sum of the individual star formation rates of all gas cells in this subhalo, but restricted to cells within the radius of 𝑉𝑚𝑎𝑥. Units Msun/yr
        "SubhaloFlag",  # Flag field indicating suitability of this subhalo for certain types of analysis. If zero, this subhalo should generally be excluded, and is not thought to be of cosmological origin.
        "SubhaloGrNr",  # Index into the Group table of the FOF host/parent of this Subhalo.
    ],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Load FoF halos, subhalos, and the header from a given snapshot.

    Args:
        basePath (str): Path to the folder containing the halos.
        snapshot (int): Index of the snapshot to use.
        halo_fields: Field to include for the FoF halos.
        subhalo_fields: Fields to include for the subhalos.

    Returns:
        (tuple[dict, dict, dict]): The loaded FoF halos, subhalos, and the header.
    """

    halos = il.groupcat.loadHalos(basePath, snapshot, fields=halo_fields)
    subhalos = il.groupcat.loadSubhalos(basePath, snapshot, fields=subhalo_fields)
    header = il.groupcat.loadHeader(basePath, snapshot)
    return halos, subhalos, header


def filter_cosmological_central_subhalos(
    halos: dict[str, Any], subhalos: dict[str, Any]
) -> dict[str, Any]:
    """
    Filter the subhalos to include only the central subhalos.

    Args:
        halos (dict[str, Any]): The FoF halos used to find central subhalos.
        subhalos (dict[str, Any]): The subhalos to filter.

    Returns:
        (dict[str, Any]): Only the central subhalos.
    """

    first = halos["GroupFirstSub"]

    # We create an array of length sub["count"] and mark those indices listed in GroupFirstSub (excluding -1).
    # GroupFirstSub points into the Subhalo table, so this is the direct way to label central subhalos.
    is_central = np.zeros(subhalos["count"], dtype=bool)

    valid = first[first >= 0]  # drop halos with no subhalos
    is_central[valid] = True

    # Combine centrality with the “useful galaxy” flag
    # SubhaloFlag == 1 indicates that the subhalo is of cosmological origin and thus suitable for analysis
    keep_filter = (subhalos["SubhaloFlag"] == 1) & is_central

    filtered_subhalos = {
        "pos_c": subhalos["SubhaloPos"][keep_filter],
        "vel_c": subhalos["SubhaloVel"][keep_filter],
        "mtype_c": subhalos["SubhaloMassType"][keep_filter],
        "grnr_c": subhalos["SubhaloGrNr"][keep_filter],
    }

    print("All subhalos:", subhalos["count"])
    print("Centrals (raw):", is_central.sum())
    print("Centrals with SubhaloFlag==1:", keep_filter.sum())

    # Optional: check how many halos have no subhalos
    print("Halos with no subhalos:", np.sum(first < 0))

    return filtered_subhalos


def filter_with_stellar_mass(
    subhalos: dict[str, Any],
    mstar_min: float = 1e10,
    mstar_max: float = 1e12,
    h: float = 0.6774,
) -> dict[str, Any]:
    """
    Filter the subhalos to include only those with stellar masses in a given range.

    Args:
        subhalos (dict[str, Any]): Subhalos to filter.
        mstar_min (float): Minimum allowed stellar mass of the subhalos [M_sol​].
        mstar_max (float): Maximum allowed stellar mass of the subhalos [M_sol​].
        h (float): Value of the reduced Hubble constant (dimensionless)

    Returns:
        (dict[str, Any]): Only the subhalos with masses in the given range.
    """

    # Stellar mass from SubhaloMassType:
    # SubhaloMassType is in 1e10 Msun/h, and index 4 corresponds to stars & wind particles.
    mstar = subhalos["mtype_c"][:, 4] * 1e10 / h  # Msun

    # Optional: dark matter mass (same units logic, index 1 is DM).
    mdm = subhalos["mtype_c"][:, 1] * 1e10 / h  # Msun

    # Build the filter.
    stellar_mass_filter = (mstar >= mstar_min) & (mstar <= mstar_max)

    # Apply it to the central sample arrays.
    filtered_subhalos = {
        "pos_s": subhalos["pos_c"][stellar_mass_filter],
        "vel_s": subhalos["vel_c"][stellar_mass_filter],
        "mstar_s": mstar[stellar_mass_filter],
        "mdm_s": mdm[stellar_mass_filter],
        "grnr_s": subhalos["grnr_c"][stellar_mass_filter],
    }

    print("\n\nCentrals before stellar-mass filter:", len(subhalos["pos_c"]))
    print("Centrals after stellar-mass filter:", len(filtered_subhalos["pos_s"]))
    print("h used:", h)
    print("Stellar mass range [Msun]:", mstar_min, "to", mstar_max)

    return filtered_subhalos


def pair_finding(
    subhalos: dict[str, Any],
    header: dict[str, Any],
    r_min: float = 500.0,
    r_max: float = 1000.0,
    h: float = 0.6774,
) -> dict[str, Any]:
    """
    Filter the subhalos to include only the pairs with a separation in a given range.

    Args:
        subhalos (dict[str, Any]): Subhalos used for pair finding.
        header (dict[str, Any]): Header with `BoxSize` key.
        r_min (float): Minimum allowed distance between the subhalos [ckpc].
        r_max (float): Maximum allowed distance between the subhalos [ckpc].
        h (float): Value of the reduced Hubble constant (dimensionless).

    Returns:
        dict[str, Any]: Subhalo pairs with separation in the given range.
    """

    box = header["BoxSize"]  # ckpc/h
    print("\n\nBoxSize [ckpc/h]:", box)

    # Distances:
    # SubhaloPos is in ckpc/h. The target window is 500–1000 kpc (physical).
    # At z=0, converting kpc -> ckpc/h is: ckpc/h = kpc * h.
    r_min = r_min * h
    r_max = r_max * h

    # Build a periodic k-d tree.
    # SciPy cKDTree supports periodic boundaries via the 'boxsize' argument. :contentReference[oaicite:3]{index=3}
    tree = cKDTree(subhalos["pos_s"], boxsize=box)

    # Get all unique index pairs (i < j) with separation <= r_max (in ckpc/h).
    pairs = tree.query_pairs(r_max, output_type="ndarray")
    print("Candidate pairs within r_max (before extra cuts):", pairs.shape[0])

    if pairs.size == 0:
        print("No pairs found.")
        return {}

    # Split indices.
    i = pairs[:, 0]
    j = pairs[:, 1]

    # Exclude pairs that live in the same FoF halo.
    # SubhaloGrNr maps each subhalo to its parent FoF halo index. :contentReference[oaicite:4]{index=4}
    diff_host = subhalos["grnr_s"][i] != subhalos["grnr_s"][j]
    i = i[diff_host]
    j = j[diff_host]

    print("Pairs after excluding same-host pairs:", i.shape[0])

    # Compute separations with the minimum-image convention.
    # This is needed to get the correct displacement vector in a periodic box.
    dr = subhalos["pos_s"][j] - subhalos["pos_s"][i]
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
    dv = subhalos["vel_s"][j] - subhalos["vel_s"][i]

    # Radial and tangential relative velocities.
    rhat = dr / dist[:, None]
    v_r = np.einsum("ij,ij->i", dv, rhat)
    v2 = np.einsum("ij,ij->i", dv, dv)
    v_t = np.sqrt(np.maximum(0.0, v2 - v_r * v_r))

    result_set = {
        "i": i,
        "j": j,
        "dr": dr,
        "dist": dist,
        "dist_kpc": dist_kpc,
        "rhat": rhat,
        "v_r": v_r,
        "v2": v2,
        "v_t": v_t,
    }

    print(
        "Separation stats [kpc]: min/median/max =",
        np.min(dist_kpc),
        np.median(dist_kpc),
        np.max(dist_kpc),
    )
    print(
        "v_r stats [km/s]: min/median/max =", np.min(v_r), np.median(v_r), np.max(v_r)
    )
    print(
        "v_t stats [km/s]: min/median/max =", np.min(v_t), np.median(v_t), np.max(v_t)
    )

    return result_set


def find_local_group_analogues(
    path: str,
    snapshot: int,
    mstar_min: float = 1e10,
    mstar_max: float = 1e12,
    r_min: float = 500.0,
    r_max: float = 1000.0,
    h: float = 0.6774,
) -> dict[str, Any]:
    """
    Find the Local Group analogue pairs by filtering based on centrality, stellar mass and separation.

    Args:
        path (str): Path to the folder containing the halos.
        snapshot (int): The index of the snapshot to use.
        mstar_min (float): Minimum allowed stellar mass of the subhalos [M_sol​].
        mstar_max (float): Maximum allowed stellar mass of the subhalos [M_sol​].
        r_min (float): Minimum allowed distance between the subhalos [ckpc].
        r_max (float): Maximum allowed distance between the subhalos [ckpc].
        h (float): Value of the reduced Hubble constant (dimensionless).

    Returns:
        (dict[str, Any]): Local group analogue pairs.
    """

    halos, subhalos, header = load_halos_subhalos_and_header_from_snapshot(
        path, snapshot
    )
    subhalos_c = filter_cosmological_central_subhalos(halos, subhalos)
    subhalos_sm = filter_with_stellar_mass(subhalos_c, mstar_min, mstar_max, h)
    return pair_finding(subhalos_sm, header, r_min, r_max, h)


# Example usage with Working directory being in the root of repository
# import os
# find_local_group_analogues(os.path.abspath("./tng300/outputs/"), 99)
