# Benchmark

Wall-clock and per-stage timings for `camera_chessboard_detector`.

All numbers are **warm averages over the full `detectChessboards()` call**
(host preprocess → map-gen → NMS → refine → score/prune → structure
recovery), measured on a single real fisheye sample resized to each preset.
The output is identical across all three pipelines and all resolutions
(**48 corners**, an 8×6 inner-corner board, on this sample); only the speed
differs.

> Status: numbers below are filled in per platform as they are measured.
> Currently only the **desktop RTX 5080** column is populated. Jetson Orin
> power modes (15 W / 30 W / 50 W / MAXN) and other desktop GPUs (RTX 30 /
> 40) are placeholders to be filled in later with the same methodology.

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

## Platform: Jetson Orin AGX  *(to be measured)*

Embedded ARM + Ampere `sm_87`; expect substantially higher numbers than
the desktop above. Fill one column per power mode using the same presets.

L4T / JetPack: _TBD_ · CUDA: _TBD_ · OpenCV: _TBD_

### Wall-clock by resolution — CUDA dense (warm avg)

| preset | resolution | MP | 15 W | 30 W | 50 W | MAXN |
|---|---:|---:|---:|---:|---:|---:|
| VGA | 640×360   | 0.23 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| SD  | 720×405   | 0.29 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| HD  | 1280×720  | 0.92 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| FHD | 1920×1080 | 2.07 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 4K  | 3840×2160 | 8.29 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Wall-clock by resolution — CUDA sep r=1 (warm avg)

| preset | resolution | MP | 15 W | 30 W | 50 W | MAXN |
|---|---:|---:|---:|---:|---:|---:|
| VGA | 640×360   | 0.23 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| SD  | 720×405   | 0.29 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| HD  | 1280×720  | 0.92 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| FHD | 1920×1080 | 2.07 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 4K  | 3840×2160 | 8.29 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Per-stage breakdown — 4K, MAXN (CUDA dense)

| stage | value |
|---|---:|
| host preprocess           | _TBD_ |
| input upload              | _TBD_ |
| Geiger map-gen            | _TBD_ |
| NMS / refine / score+prune | _TBD_ |
| structure recovery (host) | _TBD_ |
| **total wall**            | _TBD_ |

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
