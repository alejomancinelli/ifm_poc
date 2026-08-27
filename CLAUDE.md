# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Proof of concept for measuring grain volume on a conveyor belt with an **ifm
O3D303** time-of-flight camera. Current stage: reliably getting data out of the
camera. Volume computation is not implemented yet.

## Conventions

Three project skills in `.claude/skills/` define the coding standard and apply
to **all** Python in the repo. Read them before writing or reviewing code:

- `comentarios` — comments and docstrings in **Spanish**, brief, only where they
  add information. No `Args:`/`Returns:` blocks. Never comment what changed.
- `nomenclatura` — identifiers in **English** `snake_case`, `_` prefix only for
  module/class internals (never locals or parameters), every parameter
  annotated, `-> None` omitted, physical values carry their unit as a suffix
  (`timeout_ms`, `illu_temp_c`).
- `modularidad` — imports go downward through the layers, each module owns its
  formats, minimal public API.

## Commands

```powershell
# All commands assume the repo root and the checked-in venv.
.\.venv\Scripts\python.exe -m pytest                    # decoder tests; hardware tests skipped
.\.venv\Scripts\python.exe -m pytest -m hardware        # only the camera-dependent ones
.\.venv\Scripts\python.exe -m pytest tests/test_frames.py::TestDecodeFrame::test_cartesian_stays_signed

.\.venv\Scripts\python.exe scripts\probe_camera.py --ip 192.168.0.69
.\.venv\Scripts\python.exe scripts\live_view.py

.\.venv\Scripts\python.exe scripts\belt_mosaic.py --speed-m-s 3.5 --frames 20
.\.venv\Scripts\python.exe scripts\belt_mosaic.py --speed-m-s 3.5 --reference-frames 0
.\.venv\Scripts\python.exe scripts\belt_flow.py --speed-m-s 3.5 --csv run.csv
.\.venv\Scripts\python.exe scripts\belt_flow.py --speed-m-s 3.5 --record captures

.\.venv\Scripts\python.exe scripts\record_frames.py --frames 20
.\.venv\Scripts\python.exe scripts\record_frames.py --frames 0 --max-mb 500
.\.venv\Scripts\python.exe scripts\show_capture.py captures\20260825_143012 --index 3
```

`$env:O3D3XX_IP` sets the camera address for everything, and is also the switch
that enables the hardware tests. There is no build or lint step.

To rebuild the venv: `py -3.10 -m venv .venv` then
`pip install -r requirements.txt`. `o3d3xx` installs from GitHub, so `git` must
be on PATH.

## Architecture

The driver is [`ifm/o3d3xx-python`](https://github.com/ifm/o3d3xx-python),
pinned to its last upstream commit (2019) and installed from GitHub — it is not
on PyPI. It was chosen over the actively-developed `ifm3d` because `ifm3d`
targets the newer O3R platform and flags O3D support as experimental.

The camera speaks two protocols, and the layering mirrors that split:

```
settings.py   nivel 0   where the camera is; no imports from the repo
frames.py     nivel 1   pure: PCIC blobs -> numpy, validity, NaN-blanking
filters.py    nivel 1   pure: temporal median. Static scenes only; unused today.
volume.py     nivel 1   pure: reference surface, per-pixel area, integration
calibration.py nivel 1  pure: material offset fit, threshold, JSON persistence
geometry.py   nivel 1   pure: point sampling, real distances, scale verdict
profile.py    nivel 1   pure: cross-sections and their area
belt.py       nivel 1   pure: belt-fixed coordinates, frame mosaic, speed check
flow.py       nivel 1   pure: cross-section integrated over time -> volume, mass
display.py    nivel 1   pure: robust colour limits. Display only.
recorder.py   nivel 1   pure: on-disk format of a recording, disk accounting
stream.py     nivel 2   PCIC socket (port 50010)
device.py     nivel 2   XMLRPC (port 80), read-only
imager.py     nivel 2   XMLRPC edit mode (port 80) — WRITES to the camera
scripts/      nivel 5   composition roots; they wire, draw and print
```

Everything is read-only except `imager.py` and its one consumer,
`scripts/exposure_tuner.py`. Keep it that way: a script that silently reconfigures
a production camera is a bad surprise.

**In edit mode the camera stops emitting PCIC results.** This was found the hard
way — holding edit mode open while streaming hung forever. So `ImagerSession`
enters and leaves edit mode per operation and the camera runs normally in
between. A consequence: a change only survives leaving edit mode if `save()`
writes it to flash, so there is no such thing as a provisional value, and
`apply` is deliberately explicit rather than firing on every slider move.
Exposure values are rounded to integers before they are sent — the firmware
rejects the `"2998.74"` a slider drag produces.

The device permits a single session, so Vision Assistant must be closed, and the
session needs `keep_alive()` from the caller's loop or it times out.

Every PCIC connection carries a timeout (`stream.py`). Without it a silent
camera is an indistinguishable hang; `read_frame` now raises `FrameTimeoutError`
naming the two usual causes.

`frames.py` is the **sole owner of the wire formats** — pixel types, the two
valid resolutions, the confidence bit, the diagnostic blob layout. Consumers ask
for `get_valid_mask(frame)`; nobody masks bits on their own. A firmware quirk
should be a one-file fix. `NaN` is the repo-wide "no data here" value, produced
by `blank_invalid`.

`volume_roi.py` measures on the raw frame. A temporal median was tried and
reverted — it lags by half its window, which is wrong once material moves.
`filters.py` keeps it for static work like calibration; do not put it back in
the live path.

Two things about the upstream driver worth knowing, both worked around in
`frames.py`:

- Its parser reads image width and height from the PCIC chunk header and then
  **discards them**, so shape is recovered from the pixel count.
- `array.array.typecode` does survive, and it reflects the chunk's
  `pixelFormat`. That is what keeps the Cartesian buffers signed — decoding
  `x/y/z_image` as uint16 turns negative coordinates into ~65000 mm.

`scripts/` prepend the repo root to `sys.path`; the project is not installed as
a package.

## Vendored upstream code

`examples/` and `tests/manual/` are copied verbatim from upstream commit
`e57e955` and deliberately left in upstream's style (tabs, camelCase) so they
stay diffable. **The repo conventions do not apply to them.** Local changes are
marked with a `LOCAL FIX` comment — see `examples/README.md`.

Only `tests/manual/config.py` and `tests/manual/conftest.py` are ours.
`tests/manual/test_edit.py` **writes to the camera** (creates and deletes
applications) — do not run it against a device in production use.

## Domain notes

Measured on this rig: **material opacity dominates accuracy.** Soy pellets
(small, opaque) measure well; corn does not. A rigid upside-down box read its
true 74 mm, while the same box filled with corn read 56 mm — so the measurement
chain is sound and the ~18 mm is optical: light penetrates the kernel bed at
~850 nm and the reported surface sits below the real one. Tuning exposure was
tried and changed nothing, confirming the bias is optical rather than
amplitude-dependent. The fix is `calibration.py`, per material.

`min_height_mm` in a calibration deliberately serves two roles: below it the
constant-offset model stops holding (light reaches the base of a thin layer),
and it is also the threshold deciding which pixels count as material. Applying
`slope · h + offset` per pixel above the threshold yields
`slope · V + offset × covered_area` for free — the volume correction is not a
constant, and there is no separate formula for it.

The offset is measured along the camera ray, so on a tilted surface the vertical
offset is `offset / cos(θ)`. Flat lab surface under a nadir camera: exact. A
troughed belt will be under-corrected at the sides — the reference itself is
fine there, since it is captured from the empty belt and follows its shape.

`volume_roi.py` shows five panels: amplitude, Z, the ROI height map, and the two
profiles. The ROI selector lives on the **amplitude** panel because box edges
and marks are visible there and nearly invisible in Z; a dashed `Rectangle`
echoes it onto Z. All changing numbers go in the two-line figure header, never
in panel titles — titles that grow with their values collide and overflow.

For volume work, use the **Cartesian** buffers, not `distance_image`. Distance
is radial along each pixel's ray, so equal heights at different points in the
60°×45° field read differently. `z_image` already has intrinsic and extrinsic
calibration applied on-board and is a true height above the belt once the
mounting position is configured in Vision Assistant.

Volume is computed in `volume.py` as `Σ (Z_ref − Z) · pixel_area`, where
`pixel_area` is the Jacobian `|∂(X,Y)/∂(row,col)|` of the X and Y images — a far
pixel covers more surface than a near one, and the Jacobian also survives a
pixel grid rotated relative to the X/Y axes. The sum is signed on purpose so
noise cancels instead of biasing the volume upward. Each pixel is treated as a
vertical column of material, which holds for nadir mounting only.

Z sign is installation-dependent: raw, Z grows away from the lens; with the
mounting configured on the camera, Z grows upward. `compute_height_mm` takes
`is_z_up` and the scripts expose it as `--z-up`.

The metric scale comes from the camera: `x_image`/`y_image` are millimetres from
ifm's factory intrinsic calibration. Nothing here assumes a pixel pitch. A
300 mm reference on the base plane confirmed it on this rig, so **no X-Y scale
correction is applied or needed** — do not add one without re-measuring.

`profile.py` plots cross-sections against real millimetres, never pixel index:
pixels are not evenly spaced on the surface. Its `compute_area_mm2` is the same
quantity `flow.py` integrates, but taken over a single band; the belt
measurement uses the whole ROI instead, because averaging thousands of pixels
rather than three lines is most of the noise gone for free.
On a moving belt the frame is no longer the unit of measurement. At 3–4 m/s and
5 Hz the belt advances 600–800 mm per frame against a window of roughly the same
size, so an installation can land on either side of the line and summing frames
double-counts or drops material without saying which. `belt.py` tags each pixel
with `s = along − v·t`, its position **on the belt**, which is independent of the
frame that saw it: overlap averages instead of adding, and unseen belt stays an
empty cell. `flow.py` measures from the same idea without building a grid — it
integrates the mean cross-section `A = V_window / coverage_mm` against belt
travel, so what multiplies is the belt that passed and not the belt that was in
frame. `duty_cycle` (window over advance) says which regime the rig is in and is
printed on every run; below 1 the total leans on the material being even.

`coverage_mm` is measured from the live `y_image`, not assumed, as pixel step ×
pixel count — the same convention as `volume.py`'s per-pixel area, so the two
stay consistent and both follow the ROI.

Two things about `belt.py` that are not obvious. The clock is read when a frame
finishes arriving, so it carries jitter worth 3.5 mm per millisecond at 3.5 m/s;
`compute_capture_timing` snaps the instants onto the camera's fixed cadence and
counts dropped frames as whole periods. And `estimate_advance_mm` measures the
advance by correlating consecutive frames instead of trusting `--speed-m-s` —
`is_reliable` gates it, because it needs the frames to overlap and the material
to have structure, and a uniform grain bed has neither.

**The overlap bound in `estimate_advance_mm` is not a tuning knob.** The overlap
is `cells − |lag|`, so searching out to the profile length means testing lags
that leave the two views sharing a sliver, and over a sliver any lag correlates.
Caught on the bench: 9 of 66 cells in common reported 2.9 m/s on a belt running
at 1.0, at r = 0.65 — comfortably past `MIN_CORRELATION`, so the correlation
floor alone does not catch it. `MIN_OVERLAP_FRACTION` cuts the search instead, so
an unverifiable advance comes back flagged at the search edge rather than as a
number. A consequence worth knowing: at duty ≈ 1 the check will decline to
verify, which is the honest answer — raise the camera to get a real one.

`scripts/belt_mosaic.py` is the geometry test (a wrong speed, axis or direction
shows as a duplicated or torn image); `scripts/belt_flow.py` is the measurement.
The two integration paths are independent and cross-checked: on a synthetic
scene of 15.30 L the mosaic gives 15.44 L and the flow integral 15.46 L.

The reference is not for the mosaic, it is for the **height** — `Z_ref − Z` is
the only way to tell a 40 mm bed from an empty belt 40 mm closer to the lens. So
the stitch itself is judged on amplitude, which needs no reference, and
`belt_mosaic.py --reference-frames 0` skips the empty-belt phase for exactly
that: it uses `volume.build_flat_reference` (median plane of one frame), labels
the panel *relieve* rather than height, and suppresses the volume. Capturing a
reference *with* material flowing is the one thing that must not happen — the
grain surface becomes the zero, an even bed subtracts itself to nothing, and the
empty belt beside it reads as a pit. `build_flat_reference` therefore leaves
`valid_mask` entirely true: a constant plane makes no claim about which pixels
are trustworthy, and the per-frame confidence does that job.

`geometry.py` + `scripts/measure_xy.py` verify it against a known length. The
question that tool answers is not "is there an error" but "is it a uniform scale
or a position-dependent distortion" — `ScaleSummary.is_uniform` decides, using
both the spread and the correlation with radius from the image centre, because a
mean alone would hide an error that grows toward the edges. Any scale error hits
volume squared, hence `area_scale`.

**`is_reliable` gates the verdict and must be checked first.** It exists because
a real run reported a +7.9% scale error that was entirely procedure: a 70 mm
reference spans ~30 px at 352×264, so one pixel of aim is 3.3%, and the points
sat on a box *edge* — a depth discontinuity where pixels blend two surfaces
(17.5 mm of ΔZ between two points on a supposedly flat edge). `describe_scale`
suppresses the verdict entirely when the checks fail rather than printing a
number nobody should use. `has_radial_trend` compares |r| against the critical
value for the sample count; with 5 points, r = −0.69 is noise.

Error behaviour that matters when interpreting results: the volume is a signed
sum over thousands of pixels, so random noise cancels as `√N` while any
common-mode offset accumulates as `N`. A ~0.03 mm uniform bias equals the whole
random noise floor over a full frame, so thermal drift — not pixel noise —
dominates, and an empty scene never reads exactly zero. See the error budget in
README.md before "fixing" a non-zero empty reading.

`recorder.py` owns the layout of a capture folder, and it stores each blob with
the **camera's dtype** — one `.npz` per frame, one array per blob. A recorded Z
converted to float costs twice the disk, and one read back as uint16 hits the
same ~65000 mm trap `frames.py` works around; keeping the dtype avoids both.
`confidence_image` is in the default set of `scripts/record_frames.py` for the
same reason `get_valid_mask` exists: a Z without its validity mask cannot be
blanked afterwards, and it is the cheapest of the three blobs. The file name
carries the index and the local time to the millisecond so a frame that leaves
its folder still says when it was taken; the stamp is the host clock at arrival —
the O3D3xx sends no timestamp in these blobs — so `elapsed_s` in `index.csv` is
the base to measure from, and one wall-clock read anchors the run so the name and
the column cannot disagree. Milliseconds are split with `round(epoch_s * 1000)`
rather than `% 1.0`: at epoch magnitude a float64 steps in ~240 ns and truncating
loses a millisecond. `index.csv` also carries `illu_temp_c`, which is the thermal
drift of the error budget. Uncompressed size is exact arithmetic (453.8 KB per frame at 352×264 for
the default set); the compression ratio is scene-dependent, so the script
measures it instead of assuming one, and it warns when compressing eats more than
half the frame period, since it runs in the capture loop. Compression is DEFLATE
and lossless, so the only thing `--no-compress` buys is that headroom — what a
slow loop loses is whole frames, never pixel values.

`belt_flow.py --record` writes the frame from inside `_record`, next to the CSV
row, and not from `_redraw`: split across two places, one dropped frame leaves a
sample pointing at somebody else's image. It records only what was accumulated,
keeps the reference frames in their own folder because that is a different dataset
(empty belt, captured once), and stores the run parameters — speed, along-axis,
`is_z_up`, ROI — in the manifest, since the frames alone cannot be reprocessed
without them. Both folders are opened with the same `started_epoch_s` so their
file names sit on one clock. All five blobs are recorded: `x_image`/`y_image` are
what make the per-pixel area recomputable offline, and being static they compress
to almost nothing.

`tests/test_volume.py` checks the integration against analytic box volumes on a
synthetic grid, so changes to the math are verifiable without hardware.

The camera address currently lives in an env var and CLI flags. Per
`nomenclatura`, a value that varies between installations belongs in
configuration rather than a constant — if this PoC grows a `config.yaml`, the
camera address is the first thing that should move there.
