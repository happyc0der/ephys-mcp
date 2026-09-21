from .base import NeuralSource, SessionInfo
from .dandi import DandiSource
from .n1_stub import N1StubSource
from .nwb import NwbSource
from .synthetic import SyntheticSource
from .wavdir import WavDirSource

SOURCES: dict[str, dict] = {
    "synthetic": {
        "cls": SyntheticSource,
        "status": "available",
        "description": "Simulated motor-cortex units tuned to 2D cursor velocity, with ground truth.",
        "params": {"n_units": 32, "duration_s": 120.0, "noise": 1.0, "seed": 0},
    },
    "nwb": {
        "cls": NwbSource,
        "status": "available",
        "description": "A local Neurodata Without Borders (.nwb) file. Requires params.path.",
        "params": {"path": ""},
    },
    "wav_dir": {
        "cls": WavDirSource,
        "status": "available",
        "description": "Local broadband WAV: a folder of mono clips (one channel each) or one multi-channel file. "
        "Requires params.path. Set uv_per_count to report microvolts instead of ADC counts.",
        "params": {"path": "", "max_files": 64, "uv_per_count": 0.0},
    },
    "dandi": {
        "cls": DandiSource,
        "status": "available",
        "description": "Stream an NWB file from the DANDI Archive (see search_datasets, list_dataset_files). "
        "Leave path empty to pick the smallest file with behaviour.",
        "params": {"dandiset_id": "000140", "path": "", "version": ""},
    },
    "n1_stub": {
        "cls": N1StubSource,
        "status": "not_implemented",
        "description": "Contract for a live implant adapter. No public device API exists yet.",
        "params": {},
    },
}

__all__ = [
    "SOURCES",
    "DandiSource",
    "N1StubSource",
    "NeuralSource",
    "NwbSource",
    "SessionInfo",
    "SyntheticSource",
    "WavDirSource",
]
