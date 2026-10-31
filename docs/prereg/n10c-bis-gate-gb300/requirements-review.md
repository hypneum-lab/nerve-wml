# Hashed aarch64 requirements — review summary (G7 pilot)

For the PI's HITL review before access. Generated 2026-10-31 on electron-server (uv 0.13.5, metadata and wheels only, nothing installed or built), Python 3.12, `aarch64-manylinux_2_35`, `--exclude-newer 2026-10-17` (no file newer than two weeks; no direct pin had to be lowered). "latest upload" is the newest file of that version on PyPI.

**PI review: APPROVED, 2026-10-31 22:37 CET (QCM)**, under two conditions: (1) no ffmpeg
binary is downloaded for `imageio-ffmpeg` 0.3.0 — the Ubuntu ffmpeg is used through
`IMAGEIO_FFMPEG_EXE`, enforced by the lane (annex section 4); checked: the sdist ships no
binary and has no network code; (2) the rescience-nh01 condition does not apply here.

**Without an aarch64 wheel (sdist, built on the host):** gym 0.26.2, glcontext 3.1.0, glfw 1.12.0, moderngl 5.13.0, imageio-ffmpeg 0.3.0; Procgen 0.10.7 is a source build (kit PR #101, not in these files).

### `requirements-gb300-n10c-bis-gate-cuda12.txt`

| package | version | source | file | latest upload |
|---|---|---|---|---|
| cffi | 1.17.1 | aarch64 wheel | `cffi-1.17.1-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 2024-09-04 |
| cloudpickle | 3.1.2 | pure-Python wheel | `cloudpickle-3.1.2-py3-none-any.whl` | 2025-11-03 |
| filelock | 4.1.0 | pure-Python wheel | `filelock-4.1.0-py3-none-any.whl` | 2026-10-09 |
| gym-notices | 0.1.0 | pure-Python wheel | `gym_notices-0.1.0-py3-none-any.whl` | 2025-07-27 |
| imageio | 2.38.0 | pure-Python wheel | `imageio-2.38.0-py3-none-any.whl` | 2026-09-28 |
| mlx | 0.31.1 | aarch64 wheel | `mlx-0.31.1-cp312-cp312-manylinux_2_35_aarch64.whl` | 2026-03-12 |
| mlx-cuda-12 | 0.31.1 | aarch64 wheel | `mlx_cuda_12-0.31.1-py3-none-manylinux_2_35_aarch64.whl` | 2026-03-12 |
| numpy | 1.26.4 | aarch64 wheel | `numpy-1.26.4-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 2024-02-06 |
| nvidia-cublas-cu12 | 12.9.2.10 | aarch64 wheel | `nvidia_cublas_cu12-12.9.2.10-py3-none-manylinux_2_27_aarch64.whl` | 2026-04-08 |
| nvidia-cuda-nvrtc-cu12 | 12.9.86 | aarch64 wheel | `nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | 2025-06-05 |
| nvidia-cudnn-cu12 | 9.27.0.42 | aarch64 wheel | `nvidia_cudnn_cu12-9.27.0.42-py3-none-manylinux_2_27_aarch64.whl` | 2026-09-29 |
| nvidia-nccl-cu12 | 2.32.3 | aarch64 wheel | `nvidia_nccl_cu12-2.32.3-py3-none-manylinux_2_27_aarch64.whl` | 2026-09-22 |
| packaging | 26.3 | pure-Python wheel | `packaging-26.3-py3-none-any.whl` | 2026-08-04 |
| pillow | 12.3.0 | aarch64 wheel | `pillow-12.3.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl` | 2026-07-01 |
| pycparser | 3.11 | pure-Python wheel | `pycparser-3.11-py3-none-any.whl` | 2026-10-09 |
| gym3 | 0.3.3 | pure-Python wheel | `gym3-0.3.3-py3-none-any.whl` | 2020-07-10 |
| gym | 0.26.2 | **NO aarch64 wheel: sdist, built locally** | `gym-0.26.2.tar.gz` | 2022-10-04 |
| glcontext | 3.1.0 | **NO aarch64 wheel: sdist, built locally** | `glcontext-3.1.0.tar.gz` | 2026-10-07 |
| glfw | 1.12.0 | **NO aarch64 wheel: sdist, built locally** | `glfw-1.12.0.tar.gz` | 2020-07-10 |
| moderngl | 5.13.0 | **NO aarch64 wheel: sdist, built locally** | `moderngl-5.13.0.tar.gz` | 2026-10-07 |
| imageio-ffmpeg | 0.3.0 | **NO aarch64 wheel: sdist, built locally** | `imageio-ffmpeg-0.3.0.tar.gz` | 2019-04-12 |

### `requirements-gb300-n10c-bis-gate-cuda13.txt`

| package | version | source | file | latest upload |
|---|---|---|---|---|
| cffi | 1.17.1 | aarch64 wheel | `cffi-1.17.1-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 2024-09-04 |
| cloudpickle | 3.1.2 | pure-Python wheel | `cloudpickle-3.1.2-py3-none-any.whl` | 2025-11-03 |
| filelock | 4.1.0 | pure-Python wheel | `filelock-4.1.0-py3-none-any.whl` | 2026-10-09 |
| gym-notices | 0.1.0 | pure-Python wheel | `gym_notices-0.1.0-py3-none-any.whl` | 2025-07-27 |
| imageio | 2.38.0 | pure-Python wheel | `imageio-2.38.0-py3-none-any.whl` | 2026-09-28 |
| mlx | 0.31.1 | aarch64 wheel | `mlx-0.31.1-cp312-cp312-manylinux_2_35_aarch64.whl` | 2026-03-12 |
| mlx-cuda-13 | 0.31.1 | aarch64 wheel | `mlx_cuda_13-0.31.1-py3-none-manylinux_2_35_aarch64.whl` | 2026-03-12 |
| numpy | 1.26.4 | aarch64 wheel | `numpy-1.26.4-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 2024-02-06 |
| nvidia-cublas | 13.8.1.7 | aarch64 wheel | `nvidia_cublas-13.8.1.7-py3-none-manylinux_2_27_aarch64.whl` | 2026-09-28 |
| nvidia-cuda-nvrtc | 13.4.92 | aarch64 wheel | `nvidia_cuda_nvrtc-13.4.92-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | 2026-09-16 |
| nvidia-cudnn-cu13 | 9.27.0.42 | aarch64 wheel | `nvidia_cudnn_cu13-9.27.0.42-py3-none-manylinux_2_27_aarch64.whl` | 2026-09-29 |
| nvidia-nccl-cu13 | 2.32.3 | aarch64 wheel | `nvidia_nccl_cu13-2.32.3-py3-none-manylinux_2_27_aarch64.whl` | 2026-09-22 |
| packaging | 26.3 | pure-Python wheel | `packaging-26.3-py3-none-any.whl` | 2026-08-04 |
| pillow | 12.3.0 | aarch64 wheel | `pillow-12.3.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl` | 2026-07-01 |
| pycparser | 3.11 | pure-Python wheel | `pycparser-3.11-py3-none-any.whl` | 2026-10-09 |
| gym3 | 0.3.3 | pure-Python wheel | `gym3-0.3.3-py3-none-any.whl` | 2020-07-10 |
| gym | 0.26.2 | **NO aarch64 wheel: sdist, built locally** | `gym-0.26.2.tar.gz` | 2022-10-04 |
| glcontext | 3.1.0 | **NO aarch64 wheel: sdist, built locally** | `glcontext-3.1.0.tar.gz` | 2026-10-07 |
| glfw | 1.12.0 | **NO aarch64 wheel: sdist, built locally** | `glfw-1.12.0.tar.gz` | 2020-07-10 |
| moderngl | 5.13.0 | **NO aarch64 wheel: sdist, built locally** | `moderngl-5.13.0.tar.gz` | 2026-10-07 |
| imageio-ffmpeg | 0.3.0 | **NO aarch64 wheel: sdist, built locally** | `imageio-ffmpeg-0.3.0.tar.gz` | 2019-04-12 |

### `requirements-gb300-build-tools.txt`

| package | version | source | file | latest upload |
|---|---|---|---|---|
| setuptools | 84.0.0 | pure-Python wheel | `setuptools-84.0.0-py3-none-any.whl` | 2026-08-08 |
| wheel | 0.48.0 | pure-Python wheel | `wheel-0.48.0-py3-none-any.whl` | 2026-08-11 |
