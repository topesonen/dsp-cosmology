import json
from dataclasses import dataclass
import pandas as pd
import os

@dataclass
class Config:
    selection: dict         # configuration parameters for initial subhalo sample selection
    filter: dict            # configuration parameters for subhalo filtering 
    features: dict          # only active features
    all_features: dict      # all features from JSON
    target_name: str        # name of target
    target: dict            # all info of target

class ConfigLoader:
    """Loads config from JSON and provides helper functions."""
    def __init__(self, path=None):
        if path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, "config.json")
        self.path = path
        self._load_json()
        self._extract_configs()
        self._filter_features()
        self._select_target()

    def _load_json(self):
        with open(self.path, "r") as f:
            self.data = json.load(f)

    def _extract_configs(self):
        self.selection = self.data["selection_config"]
        self.filter = self.data["filter_config"]
        self.all_features = self.data["feature_info"]
        self.target_presets = self.data["target_presets"]

    def _filter_features(self):
        """Return only active features."""
        self.features = {k: v for k, v in self.all_features.items()
                         if v.get("active", True)}

    def _select_target(self):
        """Select exactly one active target."""
        active_targets = [k for k, v in self.target_presets.items()
                          if v.get("active", False)]
        if len(active_targets) != 1:
            raise ValueError(f"Expected 1 active target, got {len(active_targets)}: {active_targets}")
        self.target_name = active_targets[0]
        self.target = self.target_presets[self.target_name]

    # ---------- Helper functions ----------

    def features_table(self):
        """Return a pandas DataFrame of all features (active only by default)."""
        df = pd.DataFrame(self.all_features).T
        df["active"] = df.get("active", True)
        df = df[df["active"]]
        return df

    def all_features_table(self):
        """Return a pandas DataFrame of all features (including inactive)."""
        df = pd.DataFrame(self.all_features).T
        return df

    def targets_table(self):
        """Return a pandas DataFrame of all targets."""
        df = pd.DataFrame(self.target_presets).T
        return df

    def show_active_target(self):
        """Return the active target as DataFrame."""
        df = pd.DataFrame({self.target_name: self.target}).T
        return df
    
    def get_config(self):
        """Return a Config dataclass instance with all relevant info."""
        return Config(
            selection=self.selection,
            filter=self.filter,
            features=self.features,
            all_features=self.all_features,
            target_name=self.target_name,
            target=self.target
        )

def load_config(path=None):
    """Convenience function to quickly load ConfigLoader."""
    return ConfigLoader(path)