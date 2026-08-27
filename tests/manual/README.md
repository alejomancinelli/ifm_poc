# Manual tests

The upstream [`ifm/o3d3xx-python`](https://github.com/ifm/o3d3xx-python) test
suite, vendored verbatim from commit `e57e955`. They are `unittest.TestCase`
classes — upstream ran them under `nose`, which does not work on Python 3.10, so
we run them under `pytest` instead. No changes to the test bodies were needed.

Only `config.py` and `conftest.py` are ours.

## Running

Most of these talk to a real camera and are skipped unless you say where it is:

```powershell
# software only (parser tests) — safe anywhere
.\.venv\Scripts\python.exe -m pytest

# everything, against the camera on the bench
$env:O3D3XX_IP = "192.168.0.69"
.\.venv\Scripts\python.exe -m pytest
```

Select subsets with the `hardware` marker: `-m hardware` or `-m "not hardware"`.
The gating lives in `conftest.py` and applies only to this directory — the
decoder tests in [`../test_frames.py`](../test_frames.py) always run.

## What each file covers

| File | Camera needed | Notes |
|---|---|---|
| `test_format.py` | no | Pure PCIC chunk-parser tests against canned byte strings. Good smoke test that the install works. |
| `test_pcic.py` | yes | Opens PCIC port 50010 and asserts the protocol version is `03 01 04`. |
| `test_device.py` | yes | XMLRPC identity, session request, and the "only one session at a time" fault. |
| `test_session.py` | yes | Session heartbeat and edit-mode transitions. `test_auto_heartbeat` sleeps 40 s by design. |
| `test_edit.py` | yes | **Writes to the camera** — creates and deletes applications. |

`test_edit.py` is the one to be careful with. It creates applications and deletes
them again, so a crash mid-test can leave a stray application on the device.
Do not run it against a camera in production use.
