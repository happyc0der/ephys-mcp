from .base import NeuralSource, SessionInfo
from .dandi import DandiSource
from .live import RingBufferSource
from .lsl import LslSource
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
    "lsl": {
        "cls": LslSource,
        "status": "available",
        "description": "Subscribe to a live Lab Streaming Layer broadband stream (needs the lsl extra). "
        "Give name or type; see list_lsl_streams. Keeps the most recent buffer_s of signal.",
        "params": {"name": "", "type": "", "buffer_s": 60.0, "uv_per_count": 0.0},
    },
    "n1_stub": {
        "cls": N1StubSource,
        "status": "not_implemented",
        "description": "Contract for a live implant adapter (subclass RingBufferSource). No public device API exists yet.",
        "params": {},
    },
}

__all__ = [
    "SOURCES",
    "DandiSource",
    "LslSource",
    "N1StubSource",
    "NeuralSource",
    "NwbSource",
    "RingBufferSource",
    "SessionInfo",
    "SyntheticSource",
    "WavDirSource",
]
