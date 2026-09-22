"""Placeholder for a live implant adapter (e.g. a Neuralink N1-class device).

No vendor publishes an API for this today, so every method raises. The
docstring records what a real adapter would have to supply so that one can be
written against the same NeuralSource contract the moment an API exists.

How to write a real adapter: subclass `ephys_mcp.sources.live.RingBufferSource`
(see `lsl.py` for a complete example), open the device connection in
`__init__`, and call `self.push(samples)` from a reader thread. The base class
then provides the buffer, read_raw, threshold spike times and status.

What the device side must supply:
  * Authenticated, user-consented, local-only connection to the device relay.
  * Channel count (about 1024 electrodes) and sampling rate (about 20 kHz
    broadband). If the implant sends spike events or spike-band power instead
    of broadband, override spike_times / read_raw accordingly.
  * Amplitude scale, so results can be reported in microvolts.
  * Strictly read-only. Stimulation or device configuration must never be
    exposed through this interface.

Not affiliated with or endorsed by Neuralink Corp.
"""

from __future__ import annotations

from .base import NeuralSource

_MSG = "No public device API exists yet; see ephys_mcp/sources/n1_stub.py for the adapter contract."


class N1StubSource(NeuralSource):
    kind = "n1_stub"

    def __init__(self, *_, **__):
        raise NotImplementedError(_MSG)

    def info(self):
        raise NotImplementedError(_MSG)

    def read_raw(self, t0, t1, channels=None):
        raise NotImplementedError(_MSG)

    def spike_times(self, t0, t1):
        raise NotImplementedError(_MSG)

    def behavior(self, name, t0, t1):
        raise NotImplementedError(_MSG)
