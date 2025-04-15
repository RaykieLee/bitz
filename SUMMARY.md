# bitz-cli-gpu GPU 支持实现摘要

本文档概述了为 bitz-cli-gpu 添加 GPU 支持所做的更改和实现方式。

## 1. 文件结构更改

添加了以下新文件和目录：

- `cuda/bitz_drillx.cu`: GPU 哈希和解决方案计算的 CUDA 实现
- `cuda/bitz_drillx.h`: 公共头文件，定义了接口
- `build.rs`: 构建脚本，用于编译 CUDA 代码

## 2. 功能实现

### CUDA 实现
创建了两个主要 GPU 处理函数：
- `gpu_hash`: 使用 CUDA 进行批量哈希计算
- `gpu_solve_stages`: 使用 GPU 求解 EquiX 算法

### Rust 集成
- 添加了 `#[cfg(feature = "gpu")]` 条件编译属性以启用 GPU 支持
- 实现了 `find_hash_gpu` 函数处理 GPU 收集逻辑
- 添加了命令行参数 `--gpu` 和 `--gpu-device` 以控制 GPU 使用

## 3. 性能优化

- 使用 4096 的批处理大小，可根据实际 GPU 调整
- 使用固定内存（pinned memory）提高 CPU 和 GPU 之间的传输效率
- 利用 Rayon 并行库处理 GPU 返回的结果
- 支持多种 GPU 架构以优化性能

## 4. 命令行界面更改

修改了 `CollectArgs` 结构以支持以下新选项：
- `--gpu`: 启用 GPU 收集
- `--gpu-device [ID]`: 指定要使用的 GPU 设备 ID

## 5. 使用方法

### 编译

```bash
# 使用 GPU 特性编译
cargo build --features gpu --release
```

### 运行

```bash
# 启用 GPU 收集
./target/release/bitz collect --gpu

# 指定 GPU 设备
./target/release/bitz collect --gpu --gpu-device 1
```

## 6. 兼容性

该实现支持各种 NVIDIA GPU 架构：
- Volta (SM 70) - 如 V100
- Turing (SM 75) - 如 RTX 2080
- Ampere (SM 80, SM 86) - 如 A100、RTX 3060/3070/3080
- Ada Lovelace (SM 89) - 如 RTX 4090

## 7. 与 ore-cli-gpu 的区别

虽然参考了 ore-cli-gpu 的设计，但进行了以下改进：
- 重命名了接口和变量以遵循 bitz 命名约定
- 提高了批处理大小以增加吞吐量
- 扩展了 GPU 架构支持范围
- 改进了命令行参数和用户反馈
- 简化了安装和使用流程 