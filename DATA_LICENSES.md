# Data and licensing policy

This repository contains **no third-party data**. The only data it ships is produced by its own simulator and is dedicated to the public domain (CC0-1.0).

Rules for source adapters:

1. Real datasets are fetched or opened **by the user at runtime**. They are never committed, mirrored or redistributed here.
2. Every adapter must populate `SessionInfo.license` and `SessionInfo.citation` so the model can surface attribution to the user.
3. Datasets with no stated licence may only be read from a path the user supplies. The server must not download them on the user's behalf. The `wav_dir` source works this way: it opens local files only and reports their licence as unknown.
4. Dependencies must be permissively licensed (MIT, BSD, Apache-2.0). No GPL dependencies.
5. Code contributed here is released under CC0-1.0, so do not paste in code copied from other projects, even permissively licensed ones: their attribution terms cannot be waived by us.

The DANDI source reads each dataset's licence and citation from the archive API at open time, so what the model reports is always the archive's current statement. Curated datasets:

| Dataset | Host | Licence |
| --- | --- | --- |
| MC_Maze_Small (000140) | DANDI Archive | CC-BY-4.0 (verified from the API, 2026-09-21) |
| MC_Maze (000128), MC_RTT (000129) | DANDI Archive | reported at open time |

Trademarks: product and company names are used only descriptively. This project is not affiliated with or endorsed by any implant manufacturer.
