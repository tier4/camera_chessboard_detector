# Installing CUDA

This detector runs **out of the box on the CPU with no CUDA at all**. CUDA is
**optional** — it only adds an NVIDIA-GPU pipeline that is roughly 100–300×
faster than the CPU one (see [BENCHMARK.md](BENCHMARK.md)). If you do not have
an NVIDIA GPU, you can stop reading: just follow the build steps in the
[README](../README.md) and you are done.

This page is a step-by-step guide for people who are **not** systems
programmers. It assumes you have an NVIDIA GPU and want the fast pipeline.

---

## Contents

- [Do I even need this?](#do-i-even-need-this)
- [Which path am I on?](#which-path-am-i-on)
- [Path A — Desktop / laptop with an NVIDIA RTX GPU (Ubuntu)](#path-a--desktop--laptop-with-an-nvidia-rtx-gpu-ubuntu)
- [Path B — NVIDIA Jetson (Orin / Xavier / Nano)](#path-b--nvidia-jetson-orin--xavier--nano)
- [Checking that this project picks up CUDA](#checking-that-this-project-picks-up-cuda)
- [Troubleshooting](#troubleshooting)
- [Glossary](#glossary)

---

## Do I even need this?

| Your situation | What to do |
|---|---|
| No NVIDIA GPU (Intel/AMD graphics, a Mac, etc.) | **Skip CUDA.** Build CPU-only — see the [README](../README.md). |
| NVIDIA desktop/laptop GPU (GeForce / RTX / Quadro) | **Path A** below. |
| NVIDIA Jetson board (Orin, Xavier, Nano) | **Path B** below. CUDA is *already* on it. |

CUDA only works on **NVIDIA** GPUs. There is no CUDA for Intel or AMD graphics.

> **The two pieces you need.** People often say "install CUDA" but there are
> actually two separate things:
>
> 1. **NVIDIA driver** — lets the operating system *talk to* the GPU. Needed to
>    *run* GPU programs.
> 2. **CUDA Toolkit** — the compiler (`nvcc`) and libraries needed to *build*
>    GPU programs from source.
>
> Because you are compiling this project yourself, you need **both**. The order
> matters: driver first, toolkit second.

---

## Which path am I on?

If you are unsure whether you have an NVIDIA GPU, open a terminal and run:

```bash
lspci | grep -i nvidia
```

- **You see a line mentioning NVIDIA** (e.g. `... NVIDIA Corporation GA102 ...`)
  → you have an NVIDIA GPU. Continue with **Path A**.
- **Nothing prints** → you (probably) do not have an NVIDIA GPU. Use the
  CPU-only build.

On a Jetson this command behaves differently (the GPU is built into the chip,
not a PCI card) — if you know you are on a Jetson, go straight to **Path B**.

---

## Path A — Desktop / laptop with an NVIDIA RTX GPU (Ubuntu)

This is written for **Ubuntu 22.04 / 24.04**, the platform this project is
tested on. Other Linux distributions work too, but the exact commands differ;
follow NVIDIA's official guide for those.

> **Windows users:** the smoothest route is **WSL2** (Windows Subsystem for
> Linux). Install Ubuntu from the Microsoft Store, install a recent NVIDIA
> *Windows* driver (that is all the driver you need — do **not** install a
> Linux driver inside WSL), then follow the CUDA-Toolkit step below *inside*
> WSL. See NVIDIA's
> [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

### Step 1 — Install the NVIDIA driver

The easiest and safest way on Ubuntu is to let Ubuntu pick the right driver:

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

(Prefer a GUI? Open **"Software & Updates" → "Additional Drivers"**, select the
recommended NVIDIA driver, click **Apply**, then reboot.)

After rebooting, confirm the driver works:

```bash
nvidia-smi
```

You should see a table with your GPU name, the **Driver Version**, and a
**CUDA Version** in the top-right corner, e.g.:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 580.xx       Driver Version: 580.xx       CUDA Version: 12.8      |
|-----------------------------------------------------------------------------|
| GPU  Name        ...    | ...        | NVIDIA GeForce RTX 4070 ...           |
+-----------------------------------------------------------------------------+
```

> **What is that "CUDA Version" in `nvidia-smi`?** It is the *newest* CUDA your
> **driver** can run — **not** the toolkit you have installed. It is normal for
> it to be higher than your installed toolkit. You still need to install the
> toolkit in Step 2.

If `nvidia-smi` says *"command not found"* or *"couldn't communicate with the
NVIDIA driver"*, the driver is not installed correctly — see
[Troubleshooting](#troubleshooting) before continuing.

### Step 2 — Install the CUDA Toolkit

There are two reasonable options. **Pick one.**

#### Option 1 (simplest) — Ubuntu's packaged toolkit

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

This installs `nvcc` and puts it on your `PATH` automatically — no extra setup.
It is the least-effort path and is **fine for RTX 20/30/40 series GPUs**.

The catch: you get whatever CUDA version Ubuntu ships (often a release or two
behind). That is a problem only for **brand-new GPUs** — for example the
**RTX 50 series (Blackwell)** needs **CUDA 12.8 or newer**. If `apt` gives you
something older, the build may not be able to target your GPU. In that case use
Option 2.

#### Option 2 (recommended for newest GPUs) — NVIDIA's official repository

Use this if you have an **RTX 50-series** card, or if you simply want the
latest CUDA. Go to the official downloader and pick your exact OS:

➡️ **<https://developer.nvidia.com/cuda-downloads>**

Choose: *Linux → x86_64 → Ubuntu → your version → `deb (network)`*. The page
generates the exact copy-paste commands for you. They look roughly like this
(**do not copy these verbatim — use what the website gives you**, the version
numbers change):

```bash
# Example only — get the real commands from the link above.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit
```

> Installing `cuda-toolkit` (not the meta-package `cuda`) gives you just the
> compiler and libraries **without** pulling in another driver, which avoids
> clashing with the driver you installed in Step 1.

With Option 2 the toolkit lands in `/usr/local/cuda`, which is **not** on your
`PATH` by default. Add it once:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 3 — Verify the toolkit

```bash
nvcc --version
```

You should see something like:

```
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.8, V12.8.xx
```

If you see a version number, **you are done with CUDA.** Jump to
[Checking that this project picks up CUDA](#checking-that-this-project-picks-up-cuda).

---

## Path B — NVIDIA Jetson (Orin / Xavier / Nano)

**Good news: you do not install CUDA separately on a Jetson.** CUDA, the
driver, cuDNN, and OpenCV all come bundled in **JetPack**, NVIDIA's Jetson
software stack. Your job is just to make sure JetPack is installed and that the
toolkit is on your `PATH`.

### Step 1 — Make sure JetPack is installed

If your board was flashed with JetPack (most are out of the box, or via
**NVIDIA SDK Manager**), CUDA is already present. Check which JetPack/L4T you
have:

```bash
cat /etc/nv_tegra_release         # prints the L4T (Linux for Tegra) version
```

If JetPack is **not** installed, flash it using **SDK Manager** on a host PC,
or follow the [official Jetson getting-started guide](https://developer.nvidia.com/embedded/jetpack).
Pick a JetPack 5.x or 6.x release — both ship a CUDA 11.4+/12.x toolkit, which
is fine for this project.

### Step 2 — Put CUDA on your PATH

On Jetson the toolkit lives in a versioned folder under `/usr/local`, e.g.
`/usr/local/cuda-12.2`, with a `/usr/local/cuda` symlink. Add it to your
shell:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 3 — Verify

```bash
nvcc --version
```

A version banner means CUDA is ready.

> **`nvidia-smi` on Jetson:** older JetPack releases do **not** ship
> `nvidia-smi` (it arrives on JetPack 5+). To watch GPU usage on Jetson, the
> common tools are `tegrastats` (built in) or `jtop` (install with
> `sudo pip3 install jetson-stats`, then run `jtop`). This does not affect
> building or running the detector.

> **Performance tip (Jetson only):** Jetson boards default to a power-saving
> mode. For full speed (and for reproducible benchmark numbers) set the power
> mode and lock the clocks before measuring:
>
> ```bash
> sudo nvpmodel -m 0     # 0 is usually MAXN / max performance; check `nvpmodel -q`
> sudo jetson_clocks
> ```

---

## Checking that this project picks up CUDA

CUDA is **auto-detected** — there is no flag to turn it on. When you configure
the build, CMake prints which pipelines it will build. From the project root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Look for **one** of these lines in the output:

```
-- camera_chessboard_detector: CUDA compiler found — building CPU + CUDA pipelines
```

✅ CUDA was found — you are getting the fast pipeline.

```
-- camera_chessboard_detector: no CUDA compiler — CPU-only build
```

⚠️ CUDA was **not** found — CMake could not see `nvcc`. This almost always means
`nvcc` is not on your `PATH`; revisit Step 2/3 of your path above, open a fresh
terminal, and re-run the configure step. (If you previously configured the
project, delete the `build/` folder first so CMake re-checks: `rm -rf build`.)

Then build and run the benchmark to see the GPU in action:

```bash
cmake --build build -j
./build/camera_chessboard_detector_benchmark sample/chessboard_image.jpg 10
```

By default the build targets **your machine's own GPU** (`CMAKE_CUDA_ARCHITECTURES=native`).
If you are **cross-compiling** for a different GPU (for example building on a
desktop to deploy on a Jetson), set the target architecture explicitly:

```bash
# Jetson Orin (Ampere) is sm_87:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
```

Common compute capabilities: RTX 30-series = `86`, RTX 40-series = `89`,
RTX 50-series = `120`, Jetson Orin = `87`, Jetson Xavier = `72`. The full list
is at <https://developer.nvidia.com/cuda-gpus>.

---

## Troubleshooting

**`nvcc: command not found` (but `nvidia-smi` works).**
The *driver* is installed but the *toolkit* is not on your `PATH`. If you used
Ubuntu's `nvidia-cuda-toolkit` package, reinstall it. If you used NVIDIA's repo
or you are on Jetson, you missed the `PATH` step — re-run the
`export PATH=/usr/local/cuda/bin:$PATH` lines and open a new terminal.

**`nvidia-smi: command not found` or "couldn't communicate with the NVIDIA driver".**
The driver is missing or didn't load. On a desktop, redo Step 1 and **reboot**.
A common cause is **Secure Boot** blocking the driver module — either disable
Secure Boot in your BIOS, or complete the MOK enrollment prompt Ubuntu shows on
reboot after installing the driver. (On Jetson this command may simply not exist
on older JetPack — that's expected, see Path B.)

**CMake still says "no CUDA compiler" even though `nvcc --version` works.**
You configured the build in a terminal where `nvcc` was not yet on the `PATH`,
and CMake cached that result. Delete the build directory and configure again
from a terminal where `nvcc --version` succeeds:

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

**Build error: "no kernel image is available for execution on the device", or
the GPU pipeline produces nothing at runtime.**
The code was compiled for a different GPU architecture than the one you are
running on. Rebuild with the correct `-DCMAKE_CUDA_ARCHITECTURES=<n>` for your
card (see the table above), after `rm -rf build`.

**`CUDA driver version is insufficient for CUDA runtime version`.**
Your toolkit is newer than your driver can support. Update the NVIDIA driver
(desktop: redo Step 1 with a newer driver; Jetson: flash a newer JetPack), or
install an older CUDA toolkit that matches your driver.

**RTX 50-series (Blackwell) build fails / "unsupported gpu architecture".**
You have a CUDA toolkit older than 12.8. Blackwell (`sm_120`) needs **CUDA 12.8
or newer** — use **Path A, Option 2** to install a recent toolkit from NVIDIA's
repo.

**It built CPU-only and that's fine — I just want to confirm.**
That is a fully supported configuration. The detector produces the *same*
result on CPU and GPU; only the speed differs.

---

## Glossary

| Term | Plain-English meaning |
|---|---|
| **NVIDIA driver** | Software that lets your OS use the GPU. Needed to *run* GPU code. |
| **CUDA Toolkit** | The compiler (`nvcc`) + libraries needed to *build* GPU code. |
| **`nvidia-smi`** | A tool that reports your GPU, driver version, and live usage. |
| **`nvcc`** | The CUDA compiler. If `nvcc --version` works, the toolkit is installed. |
| **JetPack** | NVIDIA's all-in-one software stack for Jetson (includes CUDA). |
| **L4T** | "Linux for Tegra" — the Jetson operating system inside JetPack. |
| **Compute capability / `sm_XX`** | A number identifying a GPU's generation (e.g. `sm_87` = Jetson Orin), used to tell the compiler which GPU to build for. |
| **WSL2** | Windows Subsystem for Linux — runs Ubuntu inside Windows, with GPU support. |

---

For benchmark numbers per GPU, see [BENCHMARK.md](BENCHMARK.md). For building
and using the library, see the [README](../README.md).
