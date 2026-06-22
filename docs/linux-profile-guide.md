# Linux 端算子优化操作指南

## 环境信息

| 项目 | 值 |
|---|---|
| GPU | AMD Radeon 8060S (gfx1151, Strix Halo APU, 20CU, 110GB UMA VRAM) |
| ROCm | 需要 >= 6.x, 推荐 7.x |
| 仓库 | `git@github.com:runboo-fly/llama.cpp.git` (origin) |
| 上游 | `https://github.com/ggml-org/llama.cpp.git` (upstream) |

## 首次 Setup

```bash
# 1. 克隆仓库
git clone git@github.com:runboo-fly/llama.cpp.git
cd llama.cpp

# 2. 加载 profiler 脚本
source scripts/profile_ops.sh

# 3. 构建 (Ninja + HIP, 仅编译 test-backend-ops 和 llama-bench)
profile_ops_setup
```

## 第一步：找最耗时的算子

**不改代码，直接用 rocprofv3 扫真实推理，看哪些 kernel 耗时最多：**

```bash
# 载入脚本（如果还没载入）
source scripts/profile_ops.sh

# 跑真实推理 + per-kernel 耗时统计
profile_ops_real_breakdown <模型路径.gguf> "测试prompt" 20
```

这会输出 Top 20 耗时最高的 kernel 列表，直接告诉你应该优先优化哪个算子。

也可以手动跑：

```bash
rocprofv3 --stats \
    --output-dir profiles/real_breakdown \
    ./build/bin/llama-cli -m <model.gguf> -p "hello" -n 20

# 看 Top 20 耗时 kernel
cat profiles/real_breakdown/results.stats.csv | sort -t',' -k3 -rn | head -20
```

## 第二步：深入分析瓶颈算子

找到目标算子后（比如 `MUL_MAT`），用工具深入分析：

### 1. 算子级 per-kernel 耗时（轻量）

```bash
profile_ops_perf MUL_MAT
# 或指定参数过滤
profile_ops_perf MUL_MAT "type_a=f16"
```

### 2. 硬件 PMC 计数器（看 stall / cache 原因）

```bash
profile_ops_pmc MUL_MAT "type_a=f16"
```

PMC 计数器会告诉你：
- **SQ_INST_STALL_VMEM** — 显存带宽瓶颈
- **SQ_INST_STALL_LDS** — 共享内存冲突
- **SQ_INST_STALL_VALU** — 指令依赖延迟
- **TCP_TCC_READ_HIT_SUM / TCP_TCC_READ_REQ_SUM** — L1 缓存命中率

### 3. omniperf 全功能分析（推荐，数据最全）

```bash
pip install omniperf
profile_ops_omni MUL_MAT
# 分析结果
omniperf analyze -p profiles/MUL_MAT_omni/
```

## 第三步：改 kernel 源码

Kernel 源码在 `ggml/src/ggml-cuda/` 目录下，常见的优化目标：
- `fattn-tile.cuh` — Flash Attention tile kernel
- `mmvq.cu` / `mmq.cu` — 矩阵乘法量化 kernel
- `norm.cu` — RMS_NORM / LayerNorm
- `rope.cu` — 旋转位置编码
- `softmax.cu` — Softmax

## 第四步：验证改善

```bash
# 重编（增量编译，很快）
cmake --build build -j --target test-backend-ops llama-bench

# 算子级 A/B 对比
mkdir -p profiles/baseline  # 改前先把结果保存一份
cp profiles/MUL_MAT_perf/results.stats.csv profiles/baseline/

# 再跑一次 perf
profile_ops_perf MUL_MAT "type_a=f16"
# 对比 profiles/baseline/results.stats.csv vs profiles/MUL_MAT_perf/results.stats.csv
```

## 第五步：提交 & 同步到 Windows

```bash
git add ggml/src/ggml-cuda/<修改的文件>
git commit -m "ggml-cuda: optimize xxx kernel for gfx1151"
git push origin master
```

## Windows 端验证

切回 Windows 后：

```powershell
cd C:\Users\wwxq\source\repos\llama.cpp
git pull origin master
cmake --build build-amd --config Release -j

# 算子级 A/B 对比
.\build-amd\bin\test-backend-ops.exe perf -b ROCm0 -o MUL_MAT -p "type_a=f16" --output csv

# 端到端验证（最终判据）
.\build-amd\bin\llama-bench.exe -m <model.gguf> -p 512,4096,8192,32768 -n 128 -ngl 99 -fa 1 -r 5 --no-warmup -o csv
```

> **注意**: Windows 上 backend 名是 `ROCm0` 不是 `HIP`，`llama-bench` 在 `build-amd\bin\` 下。

## 工作流总结

```
Linux:   rocprofv3 --stats → 找最耗时 kernel
         omniperf profile → 看 PMC 瓶颈根因 (stall/cache miss/occupancy)
         改 kernel 源码 (.cu/.cuh)
         重编 + 验证改善
         git commit & push
           ↓
Windows: git pull
         cmake --build build-amd --config Release -j
         test-backend-ops perf A/B 对比
         llama-bench 端到端 t/s 判据
```

## 关键技巧

- **rocprofv3 --stats** 不需要改代码，直接包住 `llama-cli` 就能拿到真实推理的 per-kernel 耗时
- **omniperf** 能看到硬件级细节：为什么这个 kernel 慢（是寄存器压力、LDS 冲突、还是显存带宽瓶颈）
- 每次改完 kernel 先跑 `test-backend-ops perf` 确认算子级改善，再跑 `llama-bench` 确认端到端收益
- 算子级 `time_us` 下降但端到端 t/s 没涨 → 说明瓶颈不在这个算子，或者被其他因素掩盖了（图融合、流并发、带宽竞争）