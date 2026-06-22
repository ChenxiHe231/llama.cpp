# 开发参考 (gfx1151 / Strix Halo 算子优化)

> 这份文档是从 Claude memory 提取的关键信息，方便跨系统开发时参考。
> 不再需要依赖 Claude memory——一切都在这个仓库里。

## 环境

| 项目 | 值 |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 (Strix Halo) |
| GPU | AMD Radeon 8060S (gfx1151, RDNA3.5, 20CU, 110GB UMA VRAM) |
| 系统内存 | 128GB 物理内存, BIOS UMA 分配 ~96GB 给 GPU, ~32GB 给 Windows |
| ROCm | Windows 7.1 HIP SDK, Linux 需 >= 6.x (推荐 7.x) |
| 仓库 | `git@github.com:runboo-fly/llama.cpp.git` (origin) |
| 上游 | `https://github.com/ggml-org/llama.cpp.git` (upstream, push 已禁用) |
| 用户 | runboo-fly (Shanice He), 603479732@qq.com |

## Windows 端构建

### 主构建: `build/` (Ninja + ROCm clang + rocWMMA=OFF) — 性能基线

这是 **性能基线** 配置，rocWMMA=OFF 是关键——ON 会导致 prefill 下降 ~50%, decode(fa=1, 长上下文) 下降高达 25%。

```powershell
cd C:\Users\wwxq\source\repos\llama.cpp

# 完整构建 (首次)
cmake -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/7.1/bin/clang.exe" `
  -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/7.1/bin/clang++.exe" `
  -DGGML_HIP=ON `
  -DAMDGPU_TARGETS=gfx1151 `
  -DGGML_HIP_ROCWMMA_FATTN=OFF `
  -DGGML_OPENMP=OFF `
  -DGGML_CUDA_FA_ALL_QUANTS=ON

# 增量编译 (改 kernel 后)
cmake --build build --config Release --target llama-bench test-backend-ops -j
```

### 备选构建: `build-amd/` (VS + MSVC + rocWMMA=OFF)

```powershell
cmake -B build-amd -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-amd --config Release --target test-backend-ops llama-bench -j
```

注意：**Windows 上 backend 名是 `ROCm0`，不是 `HIP`。**

## Benchmark 命令

模型: `C:\Users\wwxq\Downloads\Qwen3.5-35B-A3B-Q4_K_M.gguf`

### Prefill sweep (不同上下文长度, fa 0/1)

```powershell
.\build\bin\llama-bench.exe `
  -m C:\Users\wwxq\Downloads\Qwen3.5-35B-A3B-Q4_K_M.gguf `
  -p 3000,5000,8000,32000,64000 -n 0 -d 0 -fa 0,1 -ngl 99 -r 5 --progress
```

### Decode sweep (不同 KV cache 大小, fa 0/1)

```powershell
.\build\bin\llama-bench.exe `
  -m C:\Users\wwxq\Downloads\Qwen3.5-35B-A3B-Q4_K_M.gguf `
  -p 0 -n 128 -d 3000,5000,8000,32000,64000 -fa 0,1 -ngl 99 -r 5 --progress
```

## 基线数据 (master efbacf8d2, rocWMMA=OFF, Qwen3.5-35B-A3B Q4_K_M)

### Prefill (pp, fa=1)

| depth | t/s |
|-------|-----|
| 64000 | 606–608 |

### Decode (tg128, fa=1)

| depth | t/s |
|-------|-----|
| 3000  | 49.30 ± 0.58 |
| 5000  | 48.61 ± 0.42 |
| 8000  | 48.07 ± 0.28 |
| 32000 | 43.58 ± 0.27 |
| 64000 | 38.31 ± 0.09 |

### Decode (tg128, fa=0) — 参考

| depth | t/s |
|-------|-----|
| 8000  | 39.95 |
| 32000 | 24.42 |
| 64000 | 12.72 |

### rocWMMA=ON 回归数据 (仅参考)

- Prefill pp=64000 fa=1: ~314 t/s (**-48%**)
- Decode fa=1 d=64000: 28.58 t/s (**-25%**)

**结论: rocWMMA=OFF 是唯一正确的基线。**

## 优化历史

### Phase 0 — `expf` → `exp2f` (暂未合并)

- 文件: `ggml/src/ggml-cuda/fattn-tile.cuh`
- 改动: 4 行, swap `expf(x)` → `exp2f(x * log2(e))`
- 效果: decode d=64000 +2.1%, 短上下文 ±0.5%
- 决策: 暂不作为独立 patch 提交, 保留 worktree 待后续合并

### Phase 1 — RDNA 配置表调优 (进行中)

- 文件: `ggml/src/ggml-cuda/fattn-tile.cuh` (约 235-310 行)
- 目标: `ggml_cuda_fattn_tile_get_config_amd_rdna` 表
- 参数格式: `GGML_CUDA_FATTN_TILE_CONFIG_CASE(DKQ, DV, ncols, nthreads, occupancy, nbatch_fa, nbatch_K)`
- 限制: nthreads≤512, occupancy≤8, nbatch_fa/K≤256

| 变体 | 改动 | 效果 | 状态 |
|------|------|------|------|
| V1 | decode: occ 8→4 | decode +0.8~+2.8% | 候选, 待合并 |
| V2 | decode: nbatch_fa 64→128 | decode -9~-15% | 已拒绝 |
| V3 | prefill: occ 3→4 | 数据不可靠 (热降频) | 待重测 |
| V4 | prefill: nbatch_fa 64→128 | — | 待开始 |

### 已知问题

- **热降频**: 连续 rebuild+bench GPU 会进入降频状态, prefill 测量受影响较大。变体之间需留 10 分钟冷却, 或用 `-r 5+` 并丢弃 stddev > 5% 的 run。

## Profiling 工具

| 工具 | 平台 | 粒度 | 说明 |
|------|------|------|------|
| `test-backend-ops perf` | Windows/Linux | 算子级 | 不改代码, 算子级 A/B 对比 |
| `llama-bench` | Windows/Linux | 端到端 | 最终 t/s 判据 |
| `rocprofv3 --stats` | Linux only | per-kernel 耗时 | 不改代码, 包住任何 binary |
| `rocprofv3 --pmc` | Linux only | 硬件计数器 | SQ stall/cache miss/带宽 |
| `omniperf` | Linux only | 全功能 PMC | `pip install omniperf` |
| `GGML_HIP_EXPORT_METRICS` | Linux only | 静态寄存器/LDS | 编译期, MSVC 不兼容 |
| hipEvent 插桩 | 需改代码 | per-kernel GPU 耗时 | dispatch loop 改几行 |

> **Windows 上没有任何 GPU profiler 工具** (无 rocprof/rocm-smi/omniperf)。详细指南见 `docs/linux-profile-guide.md`。

## rocWMMA 性能数据 (完整表格)

| depth  | fa | rocWMMA OFF | rocWMMA ON | 变化 |
|--------|----|-------------|------------|------|
| 8000   | 0  | 39.95       | 39.99      | +0.1% |
| 8000   | 1  | 48.07       | 47.21      | -1.8% |
| 32000  | 0  | 24.42       | 24.46      | +0.2% |
| 32000  | 1  | 43.58       | 35.74      | **-18%** |
| 64000  | 0  | 12.72       | 12.85      | +1.0% |
| 64000  | 1  | 38.31       | 28.58      | **-25%** |

## 内存控制 (Windows)

128GB 物理内存, BIOS UMA 分配 ~96GB 给 GPU, ~32GB 给 Windows。填充系统内存到目标水平:

```powershell
# 填充 18GB (留 ~14GB 给系统)
Start-Process -FilePath 'C:\Users\wwxq\.claude\jobs\79fc22a7\tmp\eat_ram_v3.exe' -ArgumentList '18'

# 释放
Get-Process eat_ram_v3 -ErrorAction SilentlyContinue | Stop-Process -Force

# 查看内存
Get-CimInstance Win32_OperatingSystem | Select-Object @{N='Total_GB';E={[math]::Round($_.TotalVisibleMemorySize/1MB,1)}}, @{N='Free_GB';E={[math]::Round($_.FreePhysicalMemory/1MB,1)}}
```

## Git 工作流

```bash
# 同步上游
git fetch upstream
git merge --ff-only upstream/master

# 新分支
git checkout -b feature/xxx
git push -u origin feature/xxx

# 注意: 不要 git push upstream (已被禁用)
```

## 关键 Kernel 源码位置

| 文件 | 算子 |
|------|------|
| `ggml/src/ggml-cuda/fattn-tile.cuh` | Flash Attention (tile kernel) |
| `ggml/src/ggml-cuda/mmvq.cu` | 矩阵乘法 (量化, decode) |
| `ggml/src/ggml-cuda/mmq.cu` | 矩阵乘法 (量化, prefill) |
| `ggml/src/ggml-cuda/norm.cu` | RMS_NORM / LayerNorm |
| `ggml/src/ggml-cuda/rope.cu` | 旋转位置编码 |
| `ggml/src/ggml-cuda/softmax.cu` | Softmax |

HIP backend 没有独立源码——`ggml/src/ggml-hip/CMakeLists.txt` 直接 glob 编译 `ggml-cuda/*.cu` 文件, 通过 `ggml/src/ggml-cuda/vendors/hip.h` 做 CUDA→HIP 符号映射。