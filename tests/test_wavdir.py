import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from scipy.io import wavfile

from ephys_mcp import server
from ephys_mcp.processing.spikes import match_spikes
from ephys_mcp.sources import SyntheticSource, WavDirSource

FS = 20_000
SECONDS = 5.0


@pytest.fixture(scope="module")
def sim():
    return SyntheticSource(n_units=6, duration_s=SECONDS, seed=5)


@pytest.fixture(scope="module")
def clip_dir(tmp_path_factory, sim):
    """Mono 16-bit clips, one per electrode, like a folder of exported recordings."""
    d = tmp_path_factory.mktemp("clips")
    raw = sim.read_raw(0, SECONDS)
    for ch in range(raw.shape[1]):
        n = raw.shape[0] - 100 * ch  # unequal lengths, as real exports have
        wavfile.write(d / f"clip_{ch:02d}.wav", FS, raw[:n, ch].astype(np.int16))
    (d / "notes.txt").write_text("not audio")
    return d


def test_folder_of_clips(clip_dir, sim):
    src = WavDirSource(str(clip_dir))
    info = src.info()
    assert info.n_channels == 6 and info.raw_fs_hz == FS and not info.has_sorted_spikes
    assert info.duration_s == pytest.approx(SECONDS - 500 / FS)  # cut to the shortest clip
    assert info.amplitude_unit == "ADC counts" and "not meaningful" in info.notes
    assert src.read_raw(1.0, 1.5, channels=[0, 2]).shape == (FS // 2, 2)
    detected, truth = src.spike_times(0, SECONDS), sim.spike_times(0, SECONDS)
    recall = [match_spikes(d, t)["recall"] for d, t in zip(detected, truth)]
    assert np.median(recall) > 0.85
    src.close()


def test_multichannel_file_and_calibration(tmp_path, sim):
    path = tmp_path / "array.wav"
    wavfile.write(path, FS, sim.read_raw(0, 2.0).astype(np.int16))
    src = WavDirSource(str(path), uv_per_count=0.5)
    info = src.info()
    assert info.n_channels == 6 and info.amplitude_unit == "uV" and "not meaningful" not in info.notes
    raw = src.read_raw(0, 1)
    assert raw.shape == (FS, 6) and np.abs(raw).max() <= 0.5 * 32768


def test_max_files_limits_channels(clip_dir):
    info = WavDirSource(str(clip_dir), max_files=2).info()
    assert info.n_channels == 2 and "2 of 6 clips" in info.notes


def test_bad_inputs(tmp_path, clip_dir):
    with pytest.raises(ValueError, match="no .wav files"):
        WavDirSource(str(tmp_path))
    for bad in ("https://example.com/data", str(tmp_path / "missing"), str(clip_dir / "notes.txt")):
        with pytest.raises(ValueError, match="not a folder"):
            WavDirSource(bad)
    (tmp_path / "broken.wav").write_bytes(b"not a wav file")
    with pytest.raises(ValueError, match="cannot read broken.wav"):
        WavDirSource(str(tmp_path))
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    wavfile.write(mixed / "a.wav", 20_000, np.zeros(100, np.int16))
    wavfile.write(mixed / "b.wav", 30_000, np.zeros(100, np.int16))
    with pytest.raises(ValueError, match="different sample rates"):
        WavDirSource(str(mixed))


def test_server_tools_on_wav(clip_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("EPHYS_MCP_OUTPUT_DIR", str(tmp_path))
    sid = server.open_session("wav_dir", {"path": str(clip_dir)})["session_id"]
    quality = server.get_signal_quality(sid)
    assert quality["amplitude_unit"] == "ADC counts" and quality["median_snr"] > 4
    assert "vs_ground_truth" not in server.detect_spikes(sid)
    assert server.get_firing_rates(sid)["n_units"] == 6
    assert server.plot_raster(sid, duration_s=2.0)[0]["total_spikes"] > 0
    with pytest.raises(ToolError, match="no velocity-like signal"):
        server.fit_decoder(sid)
    with pytest.raises(ToolError, match="no trial events"):
        server.get_psth(sid)
    server.close_session(sid)
