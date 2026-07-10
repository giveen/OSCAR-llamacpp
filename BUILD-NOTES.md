# OSCAR-llamacpp — Build & Run Notes (RTX 5090 / Blackwell)

This is a working copy of the OSCAR INT2-KV llama.cpp fork
(`FutureMLS-Lab/OSCAR`, branch `zhongzhu/llamacpp`), preserved at
`github.com/giveen/OSCAR-llamacpp`.

OSCAR compresses only the **KV cache** to INT2 (`q2_0`) using a calibrated,
per-layer orthogonal rotation baked into the GGUF. Model **weights stay in full
precision** (e.g. Q4_K_M). This lets you run very long contexts (tested to the
model's full 256K) for almost no extra VRAM.

---

## 1. Clone

```bash
git clone --depth 1 --branch zhongzhu/llamacpp \
  https://github.com/FutureMLS-Lab/OSCAR.git OSCAR-llamacpp
cd OSCAR-llamacpp
```

## 2. Build for NVIDIA RTX 5090 (sm_120, CUDA 13.3)

The default CUDA compiler-id probe uses `compute_52`, which CUDA 13.3's `ptxas`
no longer supports (`sm_52 is not defined`). You must force the Blackwell
architecture and point at the CUDA 13.3 toolkit explicitly.

```bash
cd /mnt/storage/Projects/OSCAR-llamacpp
mkdir -p build && cd build
export CUDA_HOME=/usr/local/cuda-13.3
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc

cmake -DGGML_CUDA=ON \
      -DGGML_CUDA_ARCHITECTURES=sm_120 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=ON \
      -DCUDACXX=/usr/local/cuda-13.3/bin/nvcc \
      ..

cmake --build . --config Release -j"$(nproc)"
```

Resulting binaries: `build/bin/llama-server`, `build/bin/llama`.

## 3. Get a model

Pre-baked OSCAR GGUF (weights Q4_K_M + rotation baked in), ~7.4 GB:

- `https://huggingface.co/Zhongzhu/OSCAR-LLAMACPP-Gemma-4-12B-it-INT2-KV`
  - `q4km-rot-kv/gemma-4-12b-it-rot-kv.gguf`

Raw rotation matrices (for self-baking other models), head_dim 256, 48 layers:

- `https://huggingface.co/Zhongzhu/OSCAR-RotationZoo` → `Gemma4-12B/`
  - `k_rotation_qqt_r_h_pbr.pt`, `v_rotation_sst_r_h_pbr.pt`

Self-bake a `-rot-kv` GGUF from a base GGUF + rotation dir:

```bash
python3 oscar-rotation/export_rot_kv_gguf.py \
  --base   /path/to/base-q4km.gguf \
  --rot-dir /path/to/Gemma4-12B \
  --out    /path/to/base-q4km-rot-kv.gguf
```

## 4. Run (the important part — Blackwell workaround)

### The crash

With `-fa on` (required, see below) and the KV cache on the GPU, the server
aborts at context build:

```
ggml-backend.cpp:898: pre-allocated tensor (cache_k_l0 (view)) in a buffer
(CUDA0) that cannot run the operation (SET_ROWS)
```

The OSCAR INT2 KV write (`SET_ROWS` into the `q2_0` cache) is incompatible with
the CUDA KV-buffer placement on Blackwell. `-fa off` does **not** help — the
server refuses because `V cache quantization requires flash_attn`.

### The fix: `-nkvo` (KV cache on CPU)

Keep `-fa on` (mandatory for q2_0 KV) and move the KV cache to system RAM with
`-nkvo`. The 2-bit KV is so small that CPU-side storage costs almost no
performance for capacity, and the SET_ROWS placement bug is avoided entirely.

```bash
cd /mnt/storage/Projects/OSCAR-llamacpp
G=/path/to/gemma-4-12b-it-rot-kv.gguf

pkill -f llama-server 2>/dev/null; sleep 2
CUDA_VISIBLE_DEVICES=0 \
GGML_CUDA_DISABLE_GRAPHS=1 \
LLAMA_KV_FUSED_FA=1 \
LLAMA_KV_NO_HADAMARD=1 \
LLAMA_KV_CLIP_RATIO=0.96 \
LLAMA_KV_HP_SINK=512 \
LLAMA_KV_HP_RECENT=2048 \
./build/bin/llama-server \
  -m "$G" \
  -fa on \
  -ngl 99 \
  -c 262144 \
  --cache-type-k q2_0 --cache-type-v q2_0 \
  -nkvo \
  --host 127.0.0.1 --port 8080
```

### Notes on the flags

| Flag | Why |
|------|-----|
| `-fa on` | **Required.** `q2_0` KV quantization needs flash attention. |
| `--cache-type-k/v q2_0` | The OSCAR INT2 KV cache. |
| `-nkvo` | **Workaround.** KV on CPU sidesteps the Blackwell `SET_ROWS` CUDA-buffer crash. |
| `LLAMA_KV_FUSED_FA=1` | Fused mixed-precision Flash-Attention kernel (OSCAR path). |
| `LLAMA_KV_NO_HADAMARD=1` | OSCAR stores plain INT2 KV (no in-quant Hadamard). |
| `LLAMA_KV_CLIP_RATIO=0.96` | Per-layer clip ratio baked during calibration. |
| `LLAMA_KV_HP_SINK=512` / `LLAMA_KV_HP_RECENT=2048` | Hybrid-prefix KV: 512 sink + 2048 recent tokens in higher precision. |
| `-c 262144` | Full native Gemma-4-12B context (model trains at 256K). |

`GGML_CUDA_DISABLE_GRAPHS=1` was also set; it did **not** by itself fix the
SET_ROWS crash (that is a graph-*reserve* failure at context build), but it is
kept to avoid runtime graph-replay issues on Blackwell. The real fix is `-nkvo`.

## 5. Verified behavior (RTX 5090, 32 GB)

- OSCAR KV engaged: `KV cache dtype: K=q2_0 V=q2_0 (n_embd_k_gqa=512, kv_size=262144)`.
- Full 256K context loads with **4 slots**, no truncation warning.
- VRAM at 256K context: **~10.4 GB** (weights Q4_K_M + 2-bit KV on CPU).
- Decode: **~45 tok/s** (CPU-KV path; GPU still does the matmuls).
- Health: `GET /health` → `{"status":"ok"}`.
- Smoke generation: prompt "The capital of France is" → "**Paris**." (coherent).

## 6. Known limitation / future work

The `-nkvo` workaround puts the KV cache on CPU, so long-context *decode*
throughput is CPU-KV-bound rather than GPU-KV-fast. A proper fix would correct
the scheduler's KV-buffer backend selection in `src/llama-kv-cache.cpp` (and the
CUDA buffer-type placement) so `SET_ROWS` into the q2_0 cache works on the CUDA
buffer — enabling GPU-KV speed at 256K.
