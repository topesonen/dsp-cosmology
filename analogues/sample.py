import illustris_python as il
from dataclasses import replace

from .data_structures import PairSet, Sample, SelectionConfig, FilterConfig
from .filters import (
    filter_by_isolation_no_third_factor,
    filter_by_isolation_no_intruder,
    filter_by_vel,
    filter_by_force_ratio,
    filter_by_tidal_dominance
)
from .pipeline import (
    AnaloguePipeline,
    load_header,
    generate_sample,
    find_pairs
)

class AnalogueSample:
    def __init__(
            self,
            base_path: str,
            selection_config: SelectionConfig,
            filter_config: FilterConfig,
            snap: int = 99,
            verbose: bool = True
        ):
        self._filter_config = filter_config
        self._selection_config = selection_config
        self._verbose = verbose

        subhalo_fields = [
            "SubhaloPos",
            "SubhaloVel",
            "SubhaloMass",
            "SubhaloMassType",
            "SubhaloFlag",
            "SubhaloGrNr",
            "SubhaloStellarPhotometrics",
        ]

        self._sub = il.groupcat.loadSubhalos(base_path, snap, fields=subhalo_fields)
        header_consts = load_header(base_path, snap, verbose)

        if self._selection_config.hubble_param is None:
            self._selection_config = replace(self._selection_config, hubble_param=header_consts["hubble_param"])
        if self._selection_config.box_side_length is None:
            self._selection_config = replace(self._selection_config, box_side_length=header_consts["box_side_length"])


        self._sample = generate_sample(
            sub=self._sub,
            basePath=base_path,
            snap=snap,
            selection_config=self._selection_config,
            verbose=verbose
        )

        self._pair_set = find_pairs(
            sub=self._sub,
            sample=self._sample,
            selection_config=self._selection_config,
            verbose=verbose
        )
        self._pipeline = AnaloguePipeline(self._pair_set)

        self._apply_filters()

    @property
    def pairs(self):
        return self._pipeline.pairs

    def _apply_filters(self):
        if self._filter_config.third_massive_factor:
            mask = filter_by_isolation_no_third_factor(
                sub=self._sub,
                pairs=self._pipeline.pairs,
                sample=self._sample,
                selection_config=self._selection_config,
                filter_config=self._filter_config
            )
            self._pipeline.apply_filter(
                " ".join([
                    f"No third subhalo more massive than factor",
                    f"x={self._filter_config.third_massive_factor}",
                    f"in same FoF group"
                ]),
                mask
            )

        if self._filter_config.third_massive_factor:
            mask = filter_by_isolation_no_intruder(
                sub=self._sub,
                pairs=self._pipeline.pairs,
                sample=self._sample,
                selection_config=self._selection_config,
                filter_config=self._filter_config
            )
            self._pipeline.apply_filter(
                " ".join([
                    f"No subhalos more massive than factor",
                    f"x={self._filter_config.intruder_factor}",
                    f"within r={self._filter_config.density_radius}",
                    f"kpc from COM"
                ]),
                mask
            )

        if any([
            self._filter_config.v_tot_max,
            self._filter_config.v_tot_min,
            self._filter_config.vt_min,            
            self._filter_config.vt_max,               
            self._filter_config.vr_min,  
            self._filter_config.vr_max,
        ]):  
            vel_mask = filter_by_vel(
                sub=self._sub,
                pairs=self._pipeline.pairs,
                sample=self._sample,
                selection_config=self._selection_config,
                filter_config=self._filter_config
            )
            self._pipeline.apply_filter(
                ", ".join([
                    f"vt in [{self._filter_config.vt_min}, {self._filter_config.vt_max}]",
                    f"vr in [{self._filter_config.vr_min}, {self._filter_config.vr_max}]", 
                    f"v_tot in [{self._filter_config.v_tot_min}, {self._filter_config.v_tot_max}]"
                ]),
                vel_mask
            )

        if self._verbose:
            self._pipeline.print_cutflow()