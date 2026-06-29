# Benchmark

Wall-clock and per-stage timings for `camera_chessboard_detector`.

All numbers are **warm averages over the full `detectChessboards()` call**
(host preprocess → map-gen → NMS → refine → score/prune → structure
recovery), measured on a single real fisheye sample resized to each preset.
The output is identical across all three pipelines and all resolutions
(**48 corners**, an 8×6 inner-corner board, on this sample); only the speed
differs.

> Status: numbers below are filled in per platform as they are measured.
> Currently the **desktop RTX 5080** and **Jetson Orin AGX (all four power
> modes)** columns are populated. Other desktop GPUs (RTX 30 / 40) are
> placeholders to be filled in later with the same methodology.

## Methodology

- Input: one 3840×2160 fisheye image, `cv::resize`d (fit-inside) to each
  preset. The sample is 16:9, so the resized presets are exact.
- Modes:
  - **CPU** — OpenCV `filter2D`-based Geiger map-gen + host pipeline.
  - **CUDA dense** — GPU map-gen via the dense convolution kernel.
  - **CUDA sep r=1** — opt-in rank-1 SVD-separable GPU map-gen
    (`ChessboardAccelerationMode::CUDA_SEPARABLE`, `separable_rank = 1`).
- Timing: one warm-up detect (absorbs CUDA context/JIT and the lazy GPU
  pipeline build), then the mean of N warm iterations (N = 10 for the
  wall-clock table). Run-to-run variance is a few ms / ~1 %.
- Reproduce: see [How to reproduce](#how-to-reproduce).

---

## Platform: desktop RTX 5080  *(measured)*

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 (Blackwell, `sm_120`, 16 GB), driver 580.126.09 |
| CPU | Intel Core Ultra 9 285K |
| RAM | 30 GiB |
| OS / toolchain | Ubuntu 24.04, CUDA 12.8, OpenCV 4.6.0, GCC 13.3, Release build |
| CUDA arch | `native` (= `sm_120`) |

### Wall-clock by resolution (warm avg, full `detectChessboards()`)

| preset | resolution | MP | CPU | CUDA dense | CUDA sep r=1 |
|---|---:|---:|---:|---:|---:|
| VGA | 640×360   | 0.23 |   80.2 ms |  6.4 ms | **5.8 ms** |
| SD  | 720×405   | 0.29 |   91.7 ms |  6.6 ms | **5.9 ms** |
| HD  | 1280×720  | 0.92 |  247.0 ms |  8.4 ms | **6.5 ms** |
| FHD | 1920×1080 | 2.07 |  558.3 ms | 11.8 ms | **7.6 ms** |
| 4K  | 3840×2160 | 8.29 | 2155.2 ms | 32.1 ms | **16.4 ms** |

All three pipelines detect the same 48 corners at every preset. The
rank-1 separable map-gen is fastest at every resolution because it cuts
the Geiger map-gen work (the dominant GPU stage at high resolution). At
low resolution the GPU map-gen shrinks to ~1 ms, so the fixed-cost host
**structure recovery** (~4.3 ms for these 48 corners, resolution-
independent) becomes the floor for the CUDA wall-clock.

### Per-stage breakdown — 4K (3840×2160)

Per-stage numbers come from the profiling build (`-DCCD_PROFILE=ON`), whose
instrumentation makes the totals run a hair higher than the Release
wall-clock table above; treat them as the per-stage *shares*.

| stage | CPU | CUDA dense | CUDA sep r=1 |
|---|---:|---:|---:|
| preprocess (GPU: BGR2GRAY + blur + normalize, incl. upload) | — | 2.1 ms | 2.0 ms |
| Geiger map-gen                                         | (in corner detect) | 23.6 ms | 8.6 ms |
| NMS (GPU)                                              | — | 0.4 ms | 0.3 ms |
| refine (GPU)                                           | — | 1.2 ms | 1.1 ms |
| score + prune (GPU + host)                            | — | 0.3 ms | 0.2 ms |
| corner detect (CPU: preprocess + conv + nms + refine + score) | 2149 ms | — | — |
| structure recovery (host)                             | 4.4 ms | 4.4 ms | 4.4 ms |
| **total wall**                                        | **2153 ms** | **32.0 ms** | **16.6 ms** |

### Per-stage breakdown — SD (720×405)

| stage | CPU | CUDA dense | CUDA sep r=1 |
|---|---:|---:|---:|
| preprocess (GPU, incl. upload) | — | 0.16 ms | 0.14 ms |
| Geiger map-gen          | (in corner detect) | 1.4 ms | 0.63 ms |
| NMS (GPU)               | — | 0.04 ms | 0.04 ms |
| refine (GPU)            | — | 0.80 ms | 0.80 ms |
| score + prune           | — | 0.09 ms | 0.09 ms |
| corner detect (CPU)     | 94 ms | — | — |
| structure recovery (host) | 4.3 ms | 4.3 ms | 4.3 ms |
| **total wall**          | **98 ms** | **6.8 ms** | **6.0 ms** |

Notes:
- The CPU front end is only instrumented at two points (the whole corner
  detect, and the structure-recovery tail); its cost is dominated by the
  six sequential `cv::filter2D` Geiger convolutions.
- On the GPU at 4K, **Geiger map-gen dominates** (~23 ms dense / ~9 ms sep);
  NMS, refine and score/prune are all on-GPU and sub-2 ms.
- The host **structure recovery (~4.3 ms) is resolution-independent** — it
  scales with the corner count (48 here), not the image size. At low
  resolution it therefore dominates the CUDA wall-clock (e.g. SD: 4.3 ms of
  the ~6.8 ms dense total).
- Preprocessing (BGR2GRAY + Gaussian blur + min-max normalize) runs on the
  GPU (~2.2 ms at 4K incl. upload), so the CPU is off the per-frame path for
  the CUDA pipeline. Note this is **not** bit-identical to OpenCV's 8-bit
  `cv::GaussianBlur`; the CUDA goldens were re-recorded for it (detection
  accuracy vs ground truth is unchanged).
- The first (cold) detect is excluded from the averages; on this host it
  costs roughly 60–100 ms extra for CUDA (context/JIT + pipeline build).

---

## Platform: Jetson Orin AGX  *(all power modes measured)*

Embedded ARM + Ampere `sm_87`. The numbers below were measured at all four
power modes — **MAXN** (`nvpmodel -m 0`), **50 W** (`nvpmodel -m 3`), **30 W**
(`nvpmodel -m 2`) and **15 W** (`nvpmodel -m 1`) — each with `jetson_clocks`
locking the CPU/GPU clocks.

| Component | Value |
|---|---|
| GPU | Jetson Orin AGX integrated Ampere (`sm_87`), shared LPDDR5 |
| CPU | 12-core Arm Cortex-A78AE |
| RAM | 64 GB (shared CPU/GPU) |
| OS / toolchain | Ubuntu 22.04.5, L4T R36.4.0 (JetPack 6.1), CUDA 12.6, OpenCV 4.5.4, GCC 11.4.0, Release build |
| Board | Connect Tech Forge carrier + Orin AGX |
| CUDA arch | `sm_87` (`-DCMAKE_CUDA_ARCHITECTURES=87` — CMake 3.22 on JetPack can't resolve `native`) |
| Clocks | `jetson_clocks`-locked — MAXN: CPU 2.2 GHz / GPU 1.3 GHz; 50 W: CPU 1.5 GHz / GPU 816 MHz; 30 W: CPU 1.7 GHz / GPU 612 MHz (8 of 12 CPU cores online); 15 W: CPU 1.1 GHz / GPU 408 MHz (4 of 12 CPU cores online) |

### Wall-clock by resolution — CUDA dense (warm avg)

| preset | resolution | MP | 15 W | 30 W | 50 W | MAXN |
|---|---:|---:|---:|---:|---:|---:|
| VGA | 640×360   | 0.23 |   81.8 ms |  46.8 ms |  35.1 ms |  23.7 ms |
| SD  | 720×405   | 0.29 |   93.5 ms |  51.8 ms |  37.0 ms |  25.0 ms |
| HD  | 1280×720  | 0.92 |  225.8 ms | 118.3 ms |  64.2 ms |  44.6 ms |
| FHD | 1920×1080 | 2.07 |  461.4 ms | 238.2 ms | 111.2 ms |  77.6 ms |
| 4K  | 3840×2160 | 8.29 | 1742.7 ms | 880.4 ms | 367.3 ms | 258.5 ms |

### Wall-clock by resolution — CUDA sep r=1 (warm avg)

| preset | resolution | MP | 15 W | 30 W | 50 W | MAXN |
|---|---:|---:|---:|---:|---:|---:|
| VGA | 640×360   | 0.23 |  47.4 ms |  29.7 ms |  28.5 ms | 19.5 ms |
| SD  | 720×405   | 0.29 |  51.3 ms |  30.7 ms |  29.2 ms | 20.1 ms |
| HD  | 1280×720  | 0.92 |  78.7 ms |  45.0 ms |  35.6 ms | 25.0 ms |
| FHD | 1920×1080 | 2.07 | 130.7 ms |  72.5 ms |  47.1 ms | 33.6 ms |
| 4K  | 3840×2160 | 8.29 | 406.2 ms | 212.7 ms | 108.7 ms | 79.8 ms |

All three pipelines detect the same 48 corners at every preset. The CPU
baseline (same methodology), VGA → 4K, is 297 / 347 / 943 / 1935 / 7166 ms
(MAXN), 379 / 437 / 1180 / 2420 / 8938 ms (30 W), 433 / 498 / 1341 / 2756 /
10141 ms (50 W) and 613 / 674 / 1818 / 3768 / 13706 ms (15 W) — note the
**30 W CPU is faster than 50 W**, because its single-core clock is higher
(1.7 vs 1.5 GHz) and the CPU front end is single-thread-bound; 15 W is slowest
on both CPU and GPU (lowest clocks of all modes). CUDA dense is ~7–28× faster
than the CPU and rank-1 separable ~13–93×; the GPU is throttled harder than the
CPU at the lower power caps, so the speed-up shrinks at 15 / 30 W. As on the
desktop, the separable map-gen is fastest at every resolution because it cuts
the dominant Geiger map-gen stage. Switching power modes needs
`sudo nvpmodel -m <id>` (see `nvpmodel -q` for the id map).

### Per-stage breakdown — 4K (3840×2160), CUDA dense (MAXN vs 50 W vs 30 W vs 15 W)

Per-stage numbers come from the profiling build (`-DCCD_PROFILE=ON`), N = 1;
treat them as the per-stage *shares*.

| stage | MAXN | 50 W | 30 W | 15 W |
|---|---:|---:|---:|---:|
| preprocess (GPU: BGR2GRAY + blur + normalize, incl. upload) | 5.6 ms | 6.9 ms | 11.2 ms | 20.2 ms |
| Geiger map-gen                                         | 231.6 ms | 331.0 ms | 837.3 ms | 1665.4 ms |
| NMS (GPU)                                              | 1.8 ms | 2.2 ms | 4.0 ms | 7.6 ms |
| refine (GPU)                                           | 2.6 ms | 3.7 ms | 5.9 ms | 10.1 ms |
| score + prune (GPU + host)                            | 1.1 ms | 1.4 ms | 2.6 ms | 4.9 ms |
| structure recovery (host)                             | 15.1 ms | 22.3 ms | 19.4 ms | 30.5 ms |
| **total wall**                                        | **258.0 ms** | **367.6 ms** | **880.5 ms** | **1739.0 ms** |

Notes:
- As on the desktop, **Geiger map-gen dominates** the GPU pipeline at 4K
  (~232 ms at MAXN, ~331 ms at 50 W, ~837 ms at 30 W, ~1665 ms at 15 W); NMS,
  refine and score/prune stay on-GPU and ≤11 ms even at 15 W.
- The host **structure recovery is resolution-independent** (~15 ms at MAXN,
  ~22 ms at 50 W, ~19 ms at 30 W, ~30 ms at 15 W) — it scales with the corner
  count (48 here), not the image size. The single-thread Cortex-A78AE makes it
  ~3–7× the desktop's 4.4 ms, so at low resolution it dominates the CUDA
  wall-clock (e.g. MAXN VGA: 15 ms of the 23.7 ms dense total).
- **Power modes are not monotonic across CPU and GPU.** The GPU clock drops
  with the power cap (1.3 GHz MAXN → 816 MHz 50 W → 612 MHz 30 W → 408 MHz
  15 W), so 4K dense rises monotonically 258 → 367 → 880 → 1743 ms (the GPU is
  throttled harder than the cap alone suggests — 30 W is ~2.4× slower than 50 W
  on the memory-bound map-gen). The **CPU is *not* monotonic with the power
  number**: by 4K CPU baseline the fastest→slowest order is MAXN (7166) → 30 W
  (8938) → 50 W (10141) → 15 W (13706), tracking the locked single-core clock
  (2.2 / 1.7 / 1.5 / 1.1 GHz) — so 30 W beats 50 W. The host structure recovery
  follows the same CPU-clock order (15 / 19 / 22 / 30 ms).
- vs desktop RTX 5080: the integrated Ampere is ~8× slower at 4K dense at MAXN
  (258 vs 32 ms), ~27× at 30 W (880 ms) and ~54× at 15 W (1743 ms).

---

## Platform: desktop RTX 30 / 40 series  *(to be measured)*

Same presets and methodology; one column per GPU.

| preset | resolution | MP | RTX 3080 dense | RTX 3080 sep r=1 | RTX 4090 dense | RTX 4090 sep r=1 |
|---|---:|---:|---:|---:|---:|---:|
| VGA | 640×360   | 0.23 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| SD  | 720×405   | 0.29 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| HD  | 1280×720  | 0.92 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| FHD | 1920×1080 | 2.07 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 4K  | 3840×2160 | 8.29 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

(Adjust the GPU columns to whatever hardware is actually measured.)

---

## How to reproduce

The benchmark tool is built with the test suite (`-DCCD_BUILD_TESTS=ON`,
the default):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Wall-clock table (bundled 4K sample, 10 warm iterations, all presets):
./build/camera_chessboard_detector_benchmark sample/chessboard_image.jpg 10

# Custom image / iteration count / single preset:
./build/camera_chessboard_detector_benchmark path/to/image.jpg 10 4K
```

Per-stage timings require a profiling build (prints map-gen / NMS / refine
/ prune / structure-recovery times to stdout):

```bash
cmake -S . -B build_prof -DCMAKE_BUILD_TYPE=Release -DCCD_PROFILE=ON
cmake --build build_prof -j
./build_prof/camera_chessboard_detector_benchmark sample/chessboard_image.jpg 1 4K
```

On Jetson, set the power mode before measuring, e.g.:

```bash
sudo nvpmodel -m <id>   # 15W / 30W / 50W / MAXN id from `nvpmodel -q`
sudo jetson_clocks      # lock clocks for repeatable numbers
```
