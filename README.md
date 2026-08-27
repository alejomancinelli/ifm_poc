# ifm O3D303 PoC

Proof of concept for measuring grain volume on a conveyor belt with an
**ifm O3D303** time-of-flight camera.

This first stage is about getting data out of the camera reliably: connect,
confirm what the device is, and look at every buffer it produces. Volume
computation comes next.

## Why `o3d3xx-python`

Two ifm libraries can talk to this camera:

- **[`ifm/o3d3xx-python`](https://github.com/ifm/o3d3xx-python)** — pure Python,
  purpose-built for the O3D3xx family. Dormant since 2019 because the O3D3xx
  protocol is frozen, not because it is broken. **This is what we use.**
- **[`ifm/ifm3d`](https://github.com/ifm/ifm3d)** — actively developed, but
  aimed at the newer O3R platform; O3D support is flagged *experimental*.

For a PoC on a legacy camera the dormant-but-exact library is the lower-risk
choice. Its 2019 `setup.py` still builds cleanly under current pip.

## Setup

Python 3.10 on Windows. The virtual environment is already created; to rebuild
it from scratch:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`o3d3xx` is not on PyPI, so `requirements.txt` installs it from GitHub pinned to
commit `e57e955` — this needs `git` on PATH.

Set the camera address once per shell instead of passing `--ip` everywhere:

```powershell
$env:O3D3XX_IP = "192.168.0.69"
```

## Usage

```powershell
# Identify the device and dump one frame of every buffer
.\.venv\Scripts\python.exe scripts\probe_camera.py --ip 192.168.0.69

# 10 frames, saved as .npz arrays plus .png previews
.\.venv\Scripts\python.exe scripts\probe_camera.py --frames 10 --save captures

# Live amplitude / distance / height view
.\.venv\Scripts\python.exe scripts\live_view.py

# Interactive volume measurement over a rectangular ROI
.\.venv\Scripts\python.exe scripts\volume_roi.py

# Tune exposure with live sliders (WRITES to the camera)
.\.venv\Scripts\python.exe scripts\exposure_tuner.py

# Calibrate the material offset against known heights
.\.venv\Scripts\python.exe scripts\calibrate_offset.py --heights 20 40 60 80 100

# Stitch N frames of a running belt into one image (checks the geometry)
.\.venv\Scripts\python.exe scripts\belt_mosaic.py --speed-m-s 3.5 --frames 20

# Same, straight onto a loaded belt: no empty-belt reference, no volume
.\.venv\Scripts\python.exe scripts\belt_mosaic.py --speed-m-s 3.5 --reference-frames 0

# Accumulate the volume passing on the belt, live
.\.venv\Scripts\python.exe scripts\belt_flow.py --speed-m-s 3.5 --csv run.csv

# Same, recording the accumulated frames for later analysis
.\.venv\Scripts\python.exe scripts\belt_flow.py --speed-m-s 3.5 --csv run.csv --record captures

# Record frames to disk and report the disk usage measured
.\.venv\Scripts\python.exe scripts\record_frames.py --frames 20

# Show a recorded frame
.\.venv\Scripts\python.exe scripts\show_capture.py captures\20260825_143012 --index 3
```

`probe_camera.py` is read-only with respect to the camera. Nothing in `scripts/`
modifies the device configuration.

## What the camera gives you

Per frame, over PCIC on port 50010:

| Blob ID | Type | Meaning |
|---|---|---|
| `normalized_amplitude_image` | uint16 | Signal strength — the practical validity signal |
| `distance_image` | uint16, mm | Radial distance along each pixel's ray |
| `x_image`, `y_image`, `z_image` | int16, mm | Cartesian point cloud, computed on-board |
| `confidence_image` | uint8 | Per-pixel status; bit 0 set = camera does not trust it |
| `diagnostic_data` | — | Illumination/frontend/CPU temperatures, evaluation time |

The **Cartesian buffers are the ones that matter for volume**. `distance_image`
is radial, so equal heights at different points in the 60°×45° field produce
different readings; `z_image` already has the intrinsic and extrinsic
calibration applied, and becomes a true height above the belt once the mounting
position is entered in Vision Assistant.

Standard resolution is 176×132 at up to 25 Hz. `ifm_poc/frames.py` resolves the
image shape from the pixel count rather than hardcoding it, so 352×264 also
works — the PCIC chunk header carries the real dimensions but the upstream
parser discards them.

## Layout

```
ifm_poc/
  settings.py     camera address defaults (env-driven)
  frames.py       pure: PCIC blobs -> numpy images, confidence bits, NaN-blanking
  filters.py      pure: temporal median (static scenes only; unused in the live path)
  volume.py       pure: reference surface, per-pixel area, height integration
  calibration.py  pure: material offset fit, threshold, JSON persistence
  geometry.py     pure: point sampling, real distances, scale verdict
  profile.py      pure: cross-sections and their area
  belt.py         pure: belt-fixed coordinates, mosaic of N frames, speed check
  flow.py         pure: cross-section integrated over time -> volume and mass
  display.py      pure: robust colour limits (display only)
  recorder.py     pure: on-disk format of a recording and its disk accounting
  stream.py       PCIC socket: open a stream, read decoded frames
  device.py       XMLRPC: read-only device identity
  imager.py       XMLRPC edit mode - the one module that WRITES to the camera
scripts/          runnable tools; only exposure_tuner.py writes to the device
examples/         upstream example scripts, vendored - see examples/README.md
tests/            test_frames.py runs without hardware
tests/manual/     upstream test suite, vendored - see tests/manual/README.md
```

`frames.py` is the only module that knows low-level formats — pixel types,
resolutions, the confidence bit, the diagnostic blob layout. Nothing else
decodes them on its own, so a firmware quirk is a one-file fix.

Code follows the repo's convention skills in `.claude/skills/`: comments and
docstrings in Spanish, identifiers in English `snake_case`, every parameter
annotated.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The decoder tests run anywhere — they build synthetic PCIC chunks and push them
through the real upstream parser. The camera-dependent ones are skipped unless
`O3D3XX_IP` is set; select them with `-m hardware`. See
[`tests/manual/README.md`](tests/manual/README.md), which flags the one test
that writes to the device.

## Volume measurement

`scripts\volume_roi.py` measures static volume — a box of grain, moved around to
check the method before anything touches a moving belt.

```
V = Σ (Z_ref − Z) · pixel_area
```

Workflow: run it, press `r` with the scene **empty** to capture the reference,
drag a rectangle on the **amplitude** panel to set the ROI, then put material
in. Keys are `r` reference, `m` log a measurement to the console, `k` toggle
compensation, `x` clear the ROI, `q` quit.

The ROI is drawn on amplitude because that is where box edges, marks and surface
texture are visible — a flat box rim is nearly invisible in Z but obvious in
amplitude. A dashed rectangle echoes the ROI on the Z panel so you can check it
against the height data too.

The reference covers the whole image, so it is captured once and the ROI can be
moved freely afterwards.

Two things this gets right that a naive version does not:

- **Per-pixel area is not constant.** The view cone makes a far pixel cover more
  surface than a near one, so the area comes from the Jacobian of the X and Y
  images, `|∂(X,Y)/∂(row,col)|`, not from an assumed uniform grid. Using the
  Jacobian rather than `|∂X/∂col|·|∂Y/∂row|` also keeps it correct when the
  pixel grid is rotated relative to the X/Y axes.
- **The sum is signed.** Noise around zero cancels instead of biasing the volume
  upward, which is what clamping negatives at zero would do.

Sign convention: by default Z grows away from the lens, so material approaching
the camera lowers Z. If the mounting position is configured on the camera, Z is
already a height that grows upward — pass `--z-up`. The window warns if the mean
height comes out strongly negative.

`volume_roi.py` measures on the **raw frame**, with no temporal filtering. A
temporal median was tried and removed: it lags by half its window, which is
invalid once material is moving. `ifm_poc/filters.py` still holds a tested
`TemporalMedian` for static work such as calibration, but nothing in the live
path uses it.

### Profiles

The bottom two panels are cross-sections through the ROI centre — one along X
(across), one along Y (along). Coloured bands on the height map show where each
cut is taken.

Height is plotted against **real millimetres, not pixel index**. Pixels are not
evenly spaced on the surface, so plotting against index would distort the
horizontal scale exactly where the view cone opens widest.

Each cut is the median of a 3-line band (`--profile-band`), not a single line —
with per-pixel noise a single line is unreadable and the median of three costs
nothing. Both panels share the height map's vertical scale, so the three read
together.

The panel titles report peak height and **cross-section area in cm²**. That last
number is the one that matters for the belt: multiplied by belt speed it is
volumetric flow, so this panel is a preview of the conveyor measurement.

### Readouts

All the changing numbers live in a two-line header at the top: volume and
compensation state on the first line, height min/median/max and the ROI's Z
range on the second. Panel titles stay short and fixed — putting the statistics
in them made them grow until they collided with each other and ran off the
figure. `m` logs the same numbers to the console, one line per measurement.

The height map uses `aspect="auto"` so a narrow ROI fills its panel instead of
becoming a sliver against one edge. Its true proportions are in the profiles,
which are plotted in millimetres.

### Colour scale

Both panels clip the top and bottom 1% of pixels when autoscaling — a handful of
bad pixels that the confidence bit did not flag would otherwise stretch the
scale far past the real range and squash everything interesting into a fraction
of the colormap.

The ROI panel rescales **every 3 seconds**, from the measured range plus a 5%
margin, rather than every frame. Per-frame rescaling makes the colours flicker
and means two frames are never directly comparable; freezing the scale between
recalculations fixes both. It also rescales immediately when the ROI moves, a
new reference is captured, or the temporal filter is still filling its window.

The margin is a **percentage of the measured span**, not a fixed number of
millimetres. A fixed 10 mm margin on data that spans ±5 mm pushes everything
into the middle third of the colormap, which reads as uniform green.

```powershell
# Fixed limits, so runs are visually comparable
.\.venv\Scripts\python.exe scripts\volume_roi.py --z-range 300 1000 --height-range 80

# Faster rescale, more headroom
.\.venv\Scripts\python.exe scripts\volume_roi.py --rescale-s 1 --scale-margin-pct 15

# Wider or narrower clipping; 0 restores raw min/max
.\.venv\Scripts\python.exe scripts\volume_roi.py --clip-pct 2
```

`--z-range MIN MAX` fixes the Z panel in mm, `--height-range MM` fixes the height
panel to ±MM; either overrides the periodic rescale. `live_view.py` takes
`--clip-pct` too. None of this touches the measurement — colour limits are
display only.

Assumption to be aware of: each pixel is treated as a vertical column of
material. That holds for a nadir-mounted camera at moderate heights. Tilt the
camera and the crest of the pile occludes what is behind it, and the volume
reads low.

## Tuning exposure

`scripts\exposure_tuner.py` is the only tool here that **writes to the camera**.

```powershell
.\.venv\Scripts\python.exe scripts\exposure_tuner.py --ip 192.168.0.69
```

**Close Vision Assistant first** — the O3D303 allows exactly one session, and a
second request fails with fault `101004`.

Sliders appear for whichever exposure parameters the active application's imager
actually exposes (`ExposureTime`, `ExposureTimeRatio`, `FrameRate`), with ranges
read from the firmware.

**Sliders do not write on their own.** Move them freely, then press `a` to
apply; the title warns while changes are pending, and `u` returns to the
original values, which are also printed on startup and on exit.

That is forced by the device, not a preference. **In edit mode the camera stops
emitting PCIC results**, and a change only survives leaving edit mode if it is
saved to flash. So applying costs a ~1 s stream interruption and is permanent —
not something to fire on every mouse movement. The tuner closes and reopens the
PCIC connection around each apply.

Three numbers drive the decision, shown live for the selected ROI:

| Readout | What it tells you |
|---|---|
| saturated % | Exposure too long — those pixels carry no usable distance |
| invalid % | Exposure too short, or the surface is too dark to measure |
| Z std dev | The actual noise that exposure choice leaves in the measurement |
| Z median | The bias — watch whether it *moves* as you drag |

That last row is the real experiment. With the scene untouched, if the Z median
shifts as you change exposure, the bias is amplitude-dependent and exposure can
fix it. If the median holds still while only the noise changes, the bias is
optical and no camera setting will remove it.

## Material offset calibration

Exposure tuning did **not** move the corn bias, which settles it: the error is
optical, not amplitude-dependent, and no camera setting will remove it. What is
left is to measure the offset and correct for it.

```powershell
# Measure 5 known heights, ignore anything under 30 mm
.\.venv\Scripts\python.exe scripts\calibrate_offset.py --heights 20 40 60 80 100 --min-height-mm 30

# Then run the measurement with the correction; `k` toggles it on and off
.\.venv\Scripts\python.exe scripts\volume_roi.py --calibration calibration.json
```

The true heights are given up front and the tool walks them in order, so the
procedure is planned rather than improvised. Press `r` with the surface empty,
drag a ROI over flat material, then `c` at each height. Each point is measured
over a temporal median of 20 frames — the scene is static, so that noise is free
to remove. The right-hand panel plots measured against true, with the identity
line for reference, so you can see the fit and where it stops holding.

`ifm_poc/calibration.py` fits `true = slope · measured + offset`. Fitting a line
rather than just averaging the offset is deliberate: **the slope tells you
whether the offset model is valid at all.** A slope near 1 means a constant
offset describes the material; a slope far from 1 means the error grows with
depth and a single number will not fix it. The report says so explicitly.

### Why there is a minimum height

`--min-height-mm` does two jobs at once, which is why it exists as one number:

1. **Below it the model is invalid.** When the layer is thinner than the light's
   penetration depth, the beam reaches the base and the bias shrinks. Including
   those points corrupts the fit.
2. **It decides which pixels count as material** when correcting volume. The
   correction is not a constant volume — it is `offset × covered_area`, so
   something must define "covered".

Both fall out of applying the correction per pixel: summing
`slope · h + offset` over pixels above the threshold gives exactly
`slope · V + offset × covered_area`, with no separate formula.

Measured on a simulated bed with an 18 mm saturating bias, threshold at 30 mm:

| | offset | slope | RMS residual |
|---|---|---|---|
| With threshold | +15.6 mm | 1.032 | **0.31 mm** |
| Without (thin layer included) | +13.1 mm | 1.071 | 1.16 mm |

The thin-layer point degrades the fit roughly fourfold. On an empty scene the
compensated volume still reads 0.000 L — the threshold stops the offset from
inventing material where there is none.

### Curved belts

Worth flagging for later, since you raised it. The reference already handles a
curved belt: it is captured from the *empty* belt, so `Z_ref` follows whatever
shape the belt has, troughed or not. That part carries over unchanged.

What does not carry over is the offset itself. Penetration happens **along the
camera ray**, so the vertical offset is `offset / cos(θ)`, where θ is the angle
between the ray and the surface normal. On a flat surface under a nadir camera
θ ≈ 0 and a scalar offset is exact — which is the lab case. On a troughed belt
the sides tilt away and this correction will **under-correct there**.

The fix, when it matters, is to estimate the surface normal from the gradient of
the Z image and scale the offset per pixel. Not implemented — it needs real
curved-belt data to validate, and guessing at it now would be speculative.

### Material matters more than settings

Measured on this rig: **soy pellets (small, opaque) measure far better than
corn.** Same geometry, same reference, same code — only the material changed.
Corn kernels are larger, waxier and more translucent at the camera's ~850 nm,
so light enters the kernel bed and scatters before returning, and the reported
surface sits below the real one.

An empty box upside-down (rigid, opaque, flat) read its true 74 mm. The same box
filled to the rim with corn read 56 mm. That ~18 mm is not a bug in the
measurement chain — the upside-down box proves the chain is sound.

## Verifying the X-Y scale

```powershell
.\.venv\Scripts\python.exe scripts\measure_xy.py --length 100
```

Click two points on a reference of known length; the tool reports what the
camera measures between them, live. `c` records a measurement, `s` prints the
report and saves it.

### Two rules for the reference, both learned the hard way

**Make it long — at least 60 pixels.** A first attempt used a 70 mm reference,
which spans only ~30 pixels at 352×264. One pixel of aiming error is then 3.3%
of the reading, so the +7.9% "scale error" it reported was 2.4 pixels of
clicking. The tool now refuses a verdict below `MIN_PIXEL_SPAN` and says how
many pixels of aim the observed error corresponds to.

**Mark the flat plane, not an edge.** That same attempt used a box edge, which
is a depth discontinuity: pixels there straddle two surfaces and report a blend
of both. The giveaway was up to 17.5 mm of Z difference between two points that
were supposed to lie on the same flat edge. Mixed pixels also smear the edge
*outward* at both ends, which is why every error came out positive rather than
scattering around zero.

Use two high-contrast marks lying flat on the base plane, far apart. `ΔZ` in the
report is the built-in check: on a flat reference it must be near zero.

**Take several placements** — centre, edges, both orientations. One measurement
cannot distinguish the two cases that matter, and they need different fixes:

| | Signature | Fix |
|---|---|---|
| Uniform scale error | Same % everywhere, low spread, no radial correlation | One scale factor |
| Position-dependent distortion | Error grows toward the field edges, high radial correlation | Per-region map; one factor would just average it |

The report states which it found rather than making you infer it — but only after
the reliability checks pass. It also weighs the radial correlation against the
critical value for the sample count: with 5 measurements you need |r| > 0.88, so
an r of −0.69 is what chance produces, not evidence of distortion.

You aim on the **amplitude** image, because marks on a flat surface are visible
there and not in Z; the measurement comes from the Cartesian buffers.

Each point is the median of a 5×5 window over 15 temporally-filtered frames, so
a single noisy pixel cannot shift a reading. It also reports the XY-projected
distance alongside the 3D one — on a flat surface they agree, and the gap
between them tells you about tilt or Z noise.

**Watch the area column.** A scale error enters volume *squared*: a +3% length
error is −5.7% in volume. The report prints the area factor next to the length
factor for that reason.

### Where the millimetres come from

`x_image` and `y_image` are real millimetres, computed **on the camera** from the
radial distance and ifm's factory intrinsic calibration (per-pixel unit vectors),
plus the extrinsic mounting parameters if they are configured. The metric scale
is not assumed anywhere in this code — that is exactly why `pixel_area_mm2`
comes from the Jacobian of X and Y rather than from a pixel pitch.

Worth validating end to end anyway: put an object of **known volume** in the ROI
and compare litres. That checks the X/Y scale and the Z scale together, which a
lateral-distance check alone does not.

### Error budget

A signed sum over thousands of pixels treats random and systematic error very
differently:

| Error type | Scales as | Full frame (23232 px, ~100 mm²/px, σ≈4 mm) |
|---|---|---|
| Random per-pixel noise | `σ·√N·A` | ≈ 0.06 L |
| Common-mode offset of 1 mm | `δ·N·A` | ≈ 2.3 L |

Random noise largely cancels; **anything that shifts the whole image uniformly
does not** — it multiplies by the full pixel count. Over a full frame a
common-mode bias of only ~0.03 mm already equals the entire random noise floor.

That is why an empty scene does not measure exactly zero, and why the residual
drifts rather than just jittering. The dominant cause is thermal: the
illumination unit and imager warm up and the measured distance drifts over tens
of minutes. `probe_camera.py` prints `illu_temp_c` from the diagnostic blob, so
the drift is directly observable.

Practical mitigations, none of which are implemented yet:

- Let the camera warm up 20–30 min before capturing the reference.
- Re-capture the reference periodically (`r`).
- Average several live frames, not just the reference — the displayed value
  already averages volume over 10 frames, which is why it is steadier than the
  per-frame number.
- Keep a strip of known-empty surface inside the frame, measure its mean height
  each frame and subtract that offset globally. This cancels common-mode drift
  almost entirely and is the single biggest available improvement.

## Measuring a moving belt

At 3–4 m/s and 5 fps the belt advances 600–800 mm between frames, while the
camera window is roughly 0.8·height along the belt — around 800 mm at 1 m. The
two numbers are the same order, so an installation can land on either side of
the line, and the naive answer (add up what each frame sees) is wrong in both
directions: with overlap it counts the same grain several times, with a gap it
misses belt entirely and never says so.

Both cases are the same problem seen from a belt-fixed coordinate. `belt.py`
tags every pixel with where it sits **on the belt**,

```
s = along − v · t
```

which does not depend on which frame saw it. Two frames that see the same grain
put it in the same cell, so it is averaged rather than added; belt that no frame
saw stays an empty cell instead of vanishing.

### Two tools

`scripts/belt_mosaic.py` captures N frames and drops them all onto one
belt-fixed grid, giving a single image of several metres of belt. It is the
geometry test: with the speed, the along-axis or the direction wrong, the
mosaic comes out duplicated, stretched or torn, and it is obvious at a glance.
It also **measures** the advance by correlating consecutive frames and prints it
next to `--speed-m-s`, so the parameter everything else rests on gets checked
against the data. That check needs a real share of the window to be common to
both frames — the overlap is `cells − lag`, so a large advance leaves the two
views sharing a sliver, and over a sliver any lag correlates well. On the bench,
9 of 66 cells in common reported 2.9 m/s for a belt running at 1.0, at r = 0.65.
The search is therefore bounded by `MIN_OVERLAP_FRACTION` and an unverifiable
advance comes back flagged, not as a number. At duty ≈ 1 expect it to decline to
verify; that is the honest answer, and raising the camera is what fixes it.

**The reference is not for the mosaic — it is for the height.** `Z_ref − Z` is
the only thing that separates a 40 mm bed from an empty belt sitting 40 mm closer
to the lens. The stitch itself is judged on the amplitude panel, which is raw
camera data, so `--reference-frames 0` skips the empty-belt phase and runs
straight onto a loaded belt: the geometry check works unchanged, the height panel
becomes *relief* against the median plane of the first frame, and the volume is
suppressed rather than printed as a plausible wrong number.

Do not capture the reference *with* material flowing. That makes the grain
surface the zero: an even bed subtracts itself to nothing and the empty belt
beside it reads as a pit.

`scripts/belt_flow.py` is the measurement. It integrates the mean cross-section
over time,

```
A = V_window / coverage_mm            [mm²]
ΔV = A · v · Δt
```

Dividing by the window length is what makes overlap harmless: what multiplies is
the belt that passed, not the belt that was in frame. Where frames do not reach
each other, the trapezoid rule fills the gap with the average of its two
neighbours, which is all that can be done without having looked. `duty_cycle`
— window length over advance — says which regime the installation is in, and is
printed on every run. Below 1 the total still comes out, but it leans on the
material being even between frames.

`coverage_mm` is measured from `y_image` on the live frame, not assumed: it is
the pixel-to-pixel step times the pixel count, so it stays consistent with the
per-pixel area `volume.py` integrates, and it follows the ROI.

Timing matters more than it looks. The clock is read when a frame finishes
arriving, so it carries network and interpreter jitter; at 3.5 m/s the belt
moves 3.5 mm per millisecond of error. `compute_capture_timing` snaps the
measured instants back onto the camera's fixed cadence, which also absorbs
dropped frames — a two-period gap is counted as two.

The two paths are independent and agree: on a synthetic scene of three known
mounds totalling 15.30 L, the mosaic integrates 15.44 L and the flow integral
15.46 L.

### From volume to weight

`--density-kg-l` is a single coefficient, and `flow.compute_density_kg_l` is the
one-liner that produces it: let material run, weigh what came out, divide by the
litres measured. It is deliberately not a table value — it absorbs the bulk
density of the material *and* whatever the optical bias in `calibration.py` did
not, so it is valid for this material on this belt and nothing else.

## Recording frames to disk

`scripts\record_frames.py` writes frames to a folder for later analysis and
prints what they cost. `--frames 20` is the test run; `--frames 0` records until
Ctrl-C, and `--max-mb` stops before the volume fills.

```powershell
.\.venv\Scripts\python.exe scripts\record_frames.py --frames 20
.\.venv\Scripts\python.exe scripts\record_frames.py --frames 0 --max-mb 500
.\.venv\Scripts\python.exe scripts\record_frames.py --blobs amplitude z --no-compress
```

One `.npz` per frame, one array per blob, plus two files that make the folder
readable without this repo: `index.csv` (frame, instant, bytes, illumination
temperature) and `session.json` (blobs, resolution, dtypes). Reading a frame back
is `np.load(path)["z_image"]`.

The file name carries the index and the local time to the millisecond —
`frame_00007_20260826_143012_345.npz` — so a frame that leaves its folder still
says when it was taken, and the names sort in capture order. That stamp is the
**host** clock when the frame finished arriving, not a camera timestamp: the
O3D3xx sends none in the blobs requested here, so it carries network and
interpreter jitter, worth 3.5 mm of belt per millisecond at 3.5 m/s. The base to
measure from is `elapsed_s` in `index.csv`, which comes off a monotonic clock —
one wall-clock read anchors the run and the rest are intervals, which also keeps
the name and the index column consistent.

Nothing is lossy at any point. `.npz` is DEFLATE, so a compressed frame reads
back byte for byte; `--no-compress` only trades disk for CPU. The one thing that
can be *lost* is a whole frame, if compression stops the loop from keeping up —
hence the warning below.

The arrays keep the **camera's dtype** — the point of not storing floats. A
`z_image` written as float is four bytes a pixel instead of two, and one decoded
as uint16 turns negative coordinates into ~65000 mm, the same trap `frames.py`
avoids on the wire. `confidence_image` is in the default set because it is the
cheapest of the three and without it the invalid pixels of a recorded Z can no
longer be blanked.

Uncompressed size is exact arithmetic — pixels × bytes per pixel:

| Blob set | 176×132 | 352×264 |
|---|---|---|
| amplitude (uint16) | 45.4 KB | 181.5 KB |
| Z (int16) | 45.4 KB | 181.5 KB |
| confidence (uint8) | 22.7 KB | 90.8 KB |
| **default: all three** | **113.5 KB** | **453.8 KB** |

Compression is on by default and roughly halves that, but the ratio depends on
the scene — a flat empty belt compresses far better than a grain bed — so the
number to trust is the one the run prints, not a table. Multiply by the frame
rate for the stream: at 5 Hz and 352×264 the raw rate is 2.2 MB/s, about
7.8 GB/h; at 25 Hz it is 11.1 MB/s, 39 GB/h. The report also gives the free
space on the volume and how long it lasts at the measured rate.

Compression runs in the capture loop, so the report prints the write time per
frame and warns when it takes more than half the frame period — past that the
camera keeps emitting and frames are lost silently. `--no-compress` trades disk
for that headroom.

`belt_flow.py --record DIR` records from inside the measurement instead, which is
the version to use when the point is to re-check a belt run later. It writes the
frames it **accumulated** — not the ones spent watching an empty belt — into
`flow/`, the reference frames into `reference/`, and puts the frame's file name in
each row of `--csv`, so a sample can be traced back to its image. Both folders
share one clock anchor, so their timestamps compare directly.

It stores all five blobs, because `x_image`/`y_image` are what let the per-pixel
area and the window be recomputed offline. That is 817 KB per frame uncompressed
at 352×264, but X and Y come from the factory calibration and never change, so
compressed they cost almost nothing — in a synthetic run the two together came to
1.1 KB of the 46 KB frame. `--record-every N` keeps one frame in N,
`--record-max-mb` stops recording (and keeps measuring) at a size, and
`--record-no-compress` takes the compression cost out of the draw loop. The run
folder is only created once a frame is actually written, and the disk report
prints on exit alongside the flow report.

`scripts\show_capture.py` reads one back and displays it — a panel per recorded
blob, amplitude in grey and the rest in viridis with the invalid pixels left
unpainted. It goes through `get_valid_mask` and `blank_invalid`, the same
functions as the live path, so a recorded frame looks exactly as `live_view.py`
would have shown it. Point it at the session folder with `--index`, or at a
single `frame_NNNNN.npz`. For PNG previews written straight from the camera
instead, `probe_camera.py --save` does that.

## Next steps toward the conveyor

1. Belt speed from an encoder instead of a parameter, so a slowing belt does not
   quietly rescale the total.
2. Belt-empty drift correction: with a troughed belt the reference follows its
   shape, but thermal drift still moves it. See the error budget above.
3. Automatic box detection, and a rotated ROI so a box that sits at an angle can
   be enclosed without clipping material at the corners.
