# ephys-mcp

An [MCP](https://modelcontextprotocol.io) server that lets an LLM analyse **intracortical (spike-level) brain-computer-interface recordings**: signal quality, spike detection, firing rates, and cursor-velocity decoding.

Existing BCI MCP servers target scalp EEG. This one targets the kind of data a high-channel-count implant produces, and defines a read-only adapter contract so a live device backend can be added when a vendor publishes an API.

> Research and education software. **Not a medical device. Not for clinical use.**
> Not affiliated with or endorsed by Neuralink Corp. or any other implant manufacturer.

## Status

v0.1, early. Working today: local NWB files, local broadband WAV recordings, streaming from the DANDI Archive, a synthetic motor-cortex source with ground truth, spike detection, quality metrics, ridge and Kalman decoders, trial-aligned PSTHs, and figures. Planned: PyPI release and registry listings.

## Install and run

```bash
uv sync
uv run ephys-mcp        # stdio transport
```

Claude Code:

```bash
claude mcp add ephys -- uv --directory /path/to/ephys-mcp run ephys-mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{ "mcpServers": { "ephys": { "command": "uv", "args": ["--directory", "/path/to/ephys-mcp", "run", "ephys-mcp"] } } }
```

Then ask, for real data: *"Find a small motor cortex dataset on DANDI, open it, and tell me how well hand velocity can be decoded."*
Or offline: *"Open a synthetic session, check signal quality, fit a Kalman decoder and show me a decoded window."*

## Data sources

| Source | What it opens |
| --- | --- |
| `synthetic` | Simulated units tuned to cursor velocity, with broadband signal and ground truth |
| `nwb` | A local `.nwb` file (`params.path`) |
| `wav_dir` | Local broadband WAV (`params.path`): a folder of mono clips, one channel each, or one multi-channel file |
| `dandi` | An NWB file streamed from the [DANDI Archive](https://dandiarchive.org) by HTTP range requests; nothing is mirrored |
| `n1_stub` | Not implemented. Documents the contract for a live implant adapter |

Dataset licence and citation come from the archive and are returned by `open_session`, so the model can attribute the data. Many datasets record only during trials; the server tracks those spans (`recorded_fraction`) and leaves the gaps out of rates and decoding instead of reading them as silence.

WAV samples carry no physical unit, so amplitudes are reported as ADC counts unless you pass `uv_per_count`; every amplitude result names its unit. Clips in a folder are separate recordings, so the server says that timing across those channels is not meaningful. Spike times from WAV are threshold crossings, not sorted units.

Reference result on MC_Maze_Small (DANDI 000140, 142 units, last 20% held out, 50 ms bins): ridge R² 0.50, Kalman R² 0.34 for hand velocity. These are simple causal linear baselines, not state of the art.

## Tools

| Tool | Purpose |
| --- | --- |
| `list_sources` | Source types and their parameters |
| `search_datasets` | Search DANDI, or list curated intracortical datasets |
| `list_dataset_files` | Licence, citation and NWB files of a DANDI dataset |
| `open_session` / `close_session` | Session lifecycle |
| `get_session_info` | Channels, rates, behaviour signals, licence, citation |
| `get_signal_quality` | Noise, SNR, dead/noisy channels |
| `detect_spikes` | Threshold crossings; precision/recall when truth exists |
| `get_firing_rates` | Population rate summary |
| `fit_decoder` | Ridge or Kalman, scored on held-out data; hyperparameters chosen inside the training split |
| `decode_window` | Decoded-vs-true preview for a window |
| `get_psth` | Firing aligned to a trial event, optionally grouped by a trial column or limited to some units |
| `plot_psth` | Figure: PSTH per group with SEM, above a unit-by-time heatmap of change from baseline |
| `plot_raster` | Figure: spike raster, unrecorded spans shaded |
| `plot_decoding` | Figure: decoded against actual behaviour, one panel per dimension |

Resource: `ephys://sessions`. Prompt: `analyze_session`.

Tools return summaries, never raw arrays, so results fit in a model's context.

Plot tools return the PNG inline, so a vision-capable model can read the figure, and also save it under `~/.cache/ephys-mcp/plots` (override with `EPHYS_MCP_OUTPUT_DIR`). Figures use a categorical palette checked for colour-blind separation, with direct labels so identity never rests on colour alone.

## Design rules

- **Read-only.** The `NeuralSource` contract has no write, stimulate or configure method. None will be added without a separate safety design.
- **Local by default.** stdio transport, no telemetry. Neural data is sensitive.
- **No bundled third-party data.** See [DATA_LICENSES.md](DATA_LICENSES.md).

## Writing a source adapter

Subclass `ephys_mcp.sources.base.NeuralSource` (`info`, `read_raw`, `spike_times`, `behavior`) and register it in `ephys_mcp/sources/__init__.py`. `sources/n1_stub.py` documents what a live implant adapter would need.

## Development

```bash
uv run pytest              # offline
uv run pytest -m network   # also streams a real file from DANDI
uv run ruff check .
```

## Licence

[CC0 1.0 Universal](LICENSE). The authors waive all copyright and related rights to the extent the law allows. Use it for anything, no attribution required. CC0 does not grant patent or trademark rights.
