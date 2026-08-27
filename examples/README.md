# Upstream examples

The example scripts from [`ifm/o3d3xx-python`](https://github.com/ifm/o3d3xx-python),
vendored from commit `e57e955` (2019-12-04) and kept close to the original so
they stay easy to diff against upstream. MIT licensed — see `UPSTREAM-LICENSE.txt`.

All of them take the camera IP as their first argument:

```powershell
.\.venv\Scripts\python.exe examples\format_grabber.py 192.168.0.69
```

| Script | What it does |
|---|---|
| `create_application.py` | **Writes to the camera.** Creates a new application, runs auto-exposure, sets free-run at 10 Hz, and makes it the active application. |
| `format_grabber.py` | Streams amplitude + distance via `FormatClient` and prints frame timing and bandwidth. The recommended client. |
| `image_grabber.py` | Same, but via the deprecated `ImageClient`, which pulls *every* buffer and so costs much more bandwidth. |
| `image_viewer.py` | matplotlib live view of amplitude and distance. |

## Local modifications

Only `image_viewer.py` was touched, and both changes are marked with a
`LOCAL FIX` comment:

- `AxesImage.get_axes()` was removed in matplotlib 3.8 → `.axes`.
- `blit=True` never repainted the titles the callback sets, and
  `cache_frame_data` grows without bound on an endless stream → `blit=False`,
  `cache_frame_data=False`.

Note the README upstream documents `FormatClient(address, format)`, but the real
signature is `FormatClient(address, port, format)`. The examples use the real one.

For our own scripts — which resolve image shape and dtype properly instead of
hardcoding 176×132 — see [`../scripts/`](../scripts/).
