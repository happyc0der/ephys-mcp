# ephys-mcp

<!-- mcp-name: io.github.happyc0der/ephys-mcp -->

An [MCP](https://modelcontextprotocol.io) server that lets an LLM analyse **intracortical (spike-level) brain-computer-interface recordings**: signal quality, spike detection, firing rates, and cursor-velocity decoding.

Existing BCI MCP servers target scalp EEG. This one targets the kind of data a high-channel-count implant produces, and defines a read-only adapter contract so a live device backend can be added when a vendor publishes an API.

> Research and education software. **Not a medical device. Not for clinical use.**
> Not affiliated with or endorsed by Neuralink Corp. or any other implant manufacturer.

## Status

v0.2, early. Working today: local NWB files, local broadband WAV recordings, live Lab Streaming Layer streams, streaming from the DANDI Archive, a synthetic motor-cortex source with ground truth, spike detection, quality metrics, ridge and Kalman decoders, trial-aligned PSTHs, spike sorting, and figures. Planned: FALCON evaluation, latent-factor models, probe geometry.

## Install and run

Needs [uv](https://docs.astral.sh/uv/). No install step: `uvx ephys-mcp` fetches the package and starts the server on stdio.

Claude Code:

```bash
claude mcp add ephys -- uvx ephys-mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{ "mcpServers": { "ephys": { "command": "uvx", "args": ["ephys-mcp"] } } }
```

From a checkout, use `uv run ephys-mcp` instead, or `uv --directory /path/to/ephys-mcp run ephys-mcp` in the configs above.

### HTTP transport

For remote clients or hosted agents, serve streamable HTTP instead of stdio:

```bash
EPHYS_MCP_TOKEN='a-long-random-secret' uvx ephys-mcp --http --host 0.0.0.0 --port 8000
```

Every request must then carry `Authorization: Bearer <token>`. The server refuses to bind to a non-loopback address without a token, and tokens must be at least 16 characters. Put TLS in front of it (a reverse proxy) before exposing it beyond a private network: the token travels in clear text otherwise. On loopback the token is optional, so `ephys-mcp --http` alone serves `http://127.0.0.1:8000/mcp` for local testing.

Then ask, for real data: *"Find a small motor cortex dataset on DANDI, open it, and tell me how well hand velocity can be decoded."*
Or offline: *"Open a synthetic session, check signal quality, fit a Kalman decoder and show me a decoded window."*

## Data sources

| Source | What it opens |
| --- | --- |
| `synthetic` | Simulated units tuned to cursor velocity, with broadband signal and ground truth |
| `nwb` | A local `.nwb` file (`params.path`) |
| `wav_dir` | Local broadband WAV (`params.path`): a folder of mono clips, one channel each, or one multi-channel file |
| `lsl` | A live [Lab Streaming Layer](https://labstreaminglayer.org) broadband stream on the local network; keeps the most recent `buffer_s` of signal. Needs `uvx --with 'ephys-mcp[lsl]' ephys-mcp` |
| `dandi` | An NWB file streamed from the [DANDI Archive](https://dandiarchive.org) by HTTP range requests; nothing is mirrored |
| `n1_stub` | Not implemented. Documents how a live implant adapter would be written on the same base as `lsl` |

Dataset licence and citation come from the archive and are returned by `open_session`, so the model can attribute the data. Many datasets record only during trials; the server tracks those spans (`recorded_fraction`) and leaves the gaps out of rates and decoding instead of reading them as silence.

WAV samples carry no physical unit, so amplitudes are reported as ADC counts unless you pass `uv_per_count`; every amplitude result names its unit. Clips in a folder are separate recordings, so the server says that timing across those channels is not meaningful. Spike times from WAV are threshold crossings, not sorted units.

Reference result on MC_Maze_Small (DANDI 000140, 142 units, last 20% held out, 50 ms bins): ridge R² 0.50, Kalman R² 0.34 for hand velocity. These are simple causal linear baselines, not state of the art.

## Tools

| Tool | Purpose |
| --- | --- |
| `list_sources` | Source types and their parameters |
| `search_datasets` | Search DANDI, or list curated intracortical datasets |
| `list_dataset_files` | Licence, citation and NWB files of a DANDI dataset |
| `list_lsl_streams` | LSL streams visible on the network |
| `get_stream_status` | For a live session: buffered span, whether data is arriving, drops |
| `open_session` / `close_session` | Session lifecycle |
| `get_session_info` | Channels, rates, behaviour signals, licence, citation |
| `get_signal_quality` | Noise, SNR, dead/noisy channels |
| `detect_spikes` | Threshold crossings; precision/recall when truth exists |
| `sort_spikes` | Spike-sort a broadband window with spikeinterface (`sort` extra); the session then uses the sorted units |
| `get_firing_rates` | Population rate summary |
| `fit_decoder` | Ridge or Kalman, scored on held-out data; hyperparameters chosen inside the training split |
| `decode_window` | Decoded-vs-true preview for a window |
| `get_psth` | Firing aligned to a trial event, optionally grouped by a trial column or limited to some units |
| `plot_psth` | Figure: PSTH per group with SEM, above a unit-by-time heatmap of change from baseline |
| `plot_raster` | Figure: spike raster, unrecorded spans shaded |
| `plot_decoding` | Figure: decoded against actual behaviour, one panel per dimension |

Resource: `ephys://sessions`. Prompt: `analyze_session`.

Tools return summaries, never raw arrays, so results fit in a model's context.

### Optional extras

| Extra | Adds | Install |
| --- | --- | --- |
| `lsl` | the `lsl` live source | `uvx --with 'ephys-mcp[lsl]' ephys-mcp` |
| `sort` | `sort_spikes` via spikeinterface's built-in sorters (spykingcircus2, tridesclous2); about 330 MB of dependencies | `uvx --with 'ephys-mcp[sort]' ephys-mcp` |

Sorting treats channels as independent electrodes because sources carry no probe geometry yet, so it suits single-electrode arrays rather than dense probes. On the simulator, spykingcircus2 recovers every unit with recall above 0.95.

Plot tools return the PNG inline, so a vision-capable model can read the figure, and also save it under `~/.cache/ephys-mcp/plots` (override with `EPHYS_MCP_OUTPUT_DIR`). Figures use a categorical palette checked for colour-blind separation, with direct labels so identity never rests on colour alone.

## Design rules

- **Read-only.** The `NeuralSource` contract has no write, stimulate or configure method. None will be added without a separate safety design.
- **Local by default.** stdio transport, no telemetry. HTTP is opt-in and token-gated. Neural data is sensitive.
- **No bundled third-party data.** See [DATA_LICENSES.md](DATA_LICENSES.md).

## Writing a source adapter

For recordings, subclass `ephys_mcp.sources.base.NeuralSource` (`info`, `read_raw`, `spike_times`, `behavior`). For a live device, subclass `ephys_mcp.sources.live.RingBufferSource` and call `push(samples)` from a reader thread; `sources/lsl.py` is a complete example in about 60 lines, and `sources/n1_stub.py` lists what an implant adapter would additionally need. Register the class in `ephys_mcp/sources/__init__.py`.

Live sessions report time as seconds since open, and only the most recent buffer is readable, so `t_start_s` and `duration_s` move forward.

## Development

```bash
uv run pytest              # offline
uv run pytest -m network   # also streams a real file from DANDI
uv run ruff check .
```

## Licence

[CC0 1.0 Universal](LICENSE). The authors waive all copyright and related rights to the extent the law allows. Use it for anything, no attribution required. CC0 does not grant patent or trademark rights.
