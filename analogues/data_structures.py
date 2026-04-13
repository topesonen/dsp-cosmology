from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration parameters for initial subhalo sample selection.

    Args:
        m_stellar_min (float): Minimum stellar mass cut in solar masses [M_sun].
        m_stellar_max (float): Maximum stellar mass cut in solar masses [M_sun].
        r_min (float): Minimum 3D distance for radial cuts in [kpc].
        r_max (float): Maximum 3D distance for radial cuts in [kpc].
        blue_threshold_gr (float): The (g - r) color index value used to 
            separate blue and red populations.
        hubble_param (float): The dimensionless Hubble parameter.
        box_side_length (float): The simulation box side length in [ckpc/h].
    """

    m_stellar_min: float    
    m_stellar_max: float    
    r_min: float    
    r_max: float
    blue_threshold_gr: float 
    hubble_param: float | None = None
    box_side_length: float | None = None


@dataclass(frozen=True)
class FilterConfig:
    """Configuration parameters for subhalo filtering.

    Args:
        v_tot_min (float): Minimum total 3D velocity magnitude in [km/s].
        v_tot_max (float): Maximum total 3D velocity magnitude in [km/s].
        vt_min (float): Minimum tangential velocity in [km/s].
        vt_max (float): Maximum tangential velocity in [km/s].
        vr_min (float): Minimum radial velocity in [km/s].
        vr_max (float): Maximum radial velocity in [km/s].
        density_radius (float): Aperture radius used for local density 
            calculations in [kpc].
        intruder_factor (float): Scaling factor used to define the influence 
            radius for nearby 'intruder' subhalos.
        third_massive_factor (float): Mass ratio threshold for the third 
            most massive subhalo in a group.
        force_ratio_min (float): Minimum force ratio threshold.
        force_ratio_max (float): Maximum force ratio threshold.
        tidally_dominant (bool): If True, only keep pairs that are tidally 
            dominant according to the force ratio criterion.
    """

    v_tot_min: float | None = 0
    v_tot_max: float | None = 500
    vt_min: float | None = 0                
    vt_max: float | None = 300               
    vr_min: float | None = 400   
    vr_max: float | None = 0  
    density_radius: float | None = 2000
    intruder_factor: float | None = 0.5
    third_massive_factor: float | None = 1.5
    force_ratio_min: float | None = None
    force_ratio_max: float | None = None
    tidally_dominant: bool | None = None


@dataclass(frozen=True)
class Sample:
    """A filtered subhalo sample with associated physical and photometric data.

    This class serves as a container for subhalo properties extracted from 
    the simulation. All arrays must have a consistent length N, 
    representing the number of subhalos in the sample.

    Args:
        keep_idx (np.ndarray): Integer indices of subhalos within the original 
            full dataset. Shape (N,).
        grnr (np.ndarray): The GroupNumber of the parent halo. Shape (N,).
        is_central (np.ndarray): Boolean-like array (1/0) indicating if the 
            subhalo is the primary member of its group. Shape (N,).
        is_blue (np.ndarray): Boolean-like array (1/0) for blue color 
            classification. Shape (N,).
        is_red (np.ndarray): Boolean-like array (1/0) for red color 
            classification. Shape (N,).
        pos (np.ndarray): Comoving spatial positions in units of [ckpc/h]. 
            Shape (N, 3).
        vel (np.ndarray): Peculiar velocities in units of [km/s]. 
            Shape (N, 3).
        m_gas (np.ndarray): Total gas mass in [10^10 M_sun/h]. Shape (N,).
        m_dark_matter (np.ndarray): Dark matter mass in [10^10 M_sun/h]. 
            Shape (N,).
        m_tracers (np.ndarray): Mass of tracer particles in [10^10 M_sun/h]. 
            Shape (N,).
        m_stellar (np.ndarray): Mass of stellar/wind particles in [10^10 M_sun/h]. 
            Shape (N,).
        m_black_hole (np.ndarray): Black hole particle mass in [10^10 M_sun/h]. 
            Shape (N,).
        m_tot (np.ndarray): Sum of all mass components in [10^10 M_sun/h]. 
            Shape (N,).
        r200c (np.ndarray): Critical radius at 200x mean density [ckpc/h]. 
            Values correspond to the parent group's central subhalo. Shape (N,).
        v200c (np.ndarray): Circular velocity at r200c [km/s]. Shape (N,).
        m200c (np.ndarray): Total mass within r200c [10^10 M_sun/h]. Shape (N,).
        u_band (np.ndarray): Absolute magnitude in the u-band (AB system). 
            Shape (N,).
        g_band (np.ndarray): Absolute magnitude in the g-band (AB system). 
            Shape (N,).
        r_band (np.ndarray): Absolute magnitude in the r-band (AB system). 
            Shape (N,).
        
    """

    keep_idx: np.ndarray
    grnr: np.ndarray
    is_central: np.ndarray
    is_blue: np.ndarray
    is_red: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    m_gas: np.ndarray
    m_dark_matter: np.ndarray
    m_tracers: np.ndarray
    m_stellar: np.ndarray
    m_black_hole: np.ndarray
    m_tot: np.ndarray
    r200c: np.ndarray
    v200c: np.ndarray
    m200c: np.ndarray
    u_band: np.ndarray
    g_band: np.ndarray
    r_band: np.ndarray
    k_band: np.ndarray
    circ: np.ndarray
    ang_mom: np.ndarray
    eigen1: np.ndarray
    eigen2: np.ndarray
    eigen3: np.ndarray


from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PairSet:
    """A collection of subhalo pairs with derived kinematics and classifications.

    This class stores indices pointing to a primary `Sample` object and provides
    pre-computed physical relationships between the paired subhalos. All arrays 
    must share a consistent length M (number of pairs).

    Args:
        i (np.ndarray): Indices of the first subhalo in the pair. Shape (M,).
        j (np.ndarray): Indices of the second subhalo in the pair. Shape (M,).
        have_same_host (np.ndarray): Boolean-like (1/0) indicating if both 
            subhalos belong to the same Friends-of-Friends (FoF) group. Shape (M,).
        is_tidally_dominant (np.ndarray): Boolean-like (1/0) indicating if the 
            pair's mutual gravitational interaction dominates over the 
            background tidal field. Shape (M,).
        is_blue_blue (np.ndarray): Boolean-like (1/0) indicating if both 
            members are classified as blue galaxies. Shape (M,).
        is_red_red (np.ndarray): Boolean-like (1/0) indicating if both 
            members are classified as red galaxies. Shape (M,).
        is_blue_red (np.ndarray): Boolean-like (1/0) indicating if the pair
            is a mix of a blue and a red galaxy. Shape (M,).
        separation (np.ndarray): Physical 3D separation between members in [kpc]. 
            Shape (M,).
        vr (np.ndarray): Radial component of the relative velocity in [km/s]. 
            Shape (M,).
        vt (np.ndarray): Tangential (transverse) component of the relative 
            velocity in [km/s]. Shape (M,).
        force_ratio (np.ndarray): Dimensionless ratio of the mutual 
            gravitational force to the external tidal force. Shape (M,).
        log_tot_virial_mass (np.ndarray): Log10 of the combined virial mass 
            of the pair in [M_sun]. Shape (M,).
        log_mass_ratio (np.ndarray): Log10 of the mass ratio (M_smaller / M_larger). 
            Results in values <= 0. Shape (M,).
    """

    i: np.ndarray
    j: np.ndarray
    have_same_host: np.ndarray
    is_tidally_dominant: np.ndarray
    is_blue_blue: np.ndarray
    is_red_red: np.ndarray
    is_blue_red: np.ndarray
    separation: np.ndarray
    vr: np.ndarray
    vt: np.ndarray
    force_ratio: np.ndarray
    log_tot_virial_mass: np.ndarray
    log_mass_ratio: np.ndarray

    def apply_mask(self, mask: np.ndarray) -> 'PairSet':
        """Filters the PairSet using a boolean mask.

        Args:
            mask (np.ndarray): A boolean array of length M where True 
                indicates a pair to keep.

        Returns:
            PairSet: A new instance containing only the masked data.
        """
        return PairSet(
            i=self.i[mask],
            j=self.j[mask],
            have_same_host=self.have_same_host[mask],
            is_tidally_dominant=self.is_tidally_dominant[mask],
            is_blue_blue=self.is_blue_blue[mask],
            is_red_red=self.is_red_red[mask],
            is_blue_red=self.is_blue_red[mask],
            separation=self.separation[mask],
            vr=self.vr[mask],
            vt=self.vt[mask],
            force_ratio=self.force_ratio[mask],
            log_tot_virial_mass=self.log_tot_virial_mass[mask],
            log_mass_ratio=self.log_mass_ratio[mask]
        )