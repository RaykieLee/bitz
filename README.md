# BITZ Collector with GPU Support

A command line interface for BITZ cryptocurrency collecting with GPU acceleration.

## 📦 Install

To install the CLI with GPU support, use [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html):

```sh
# 编译并安装 GPU 版本
cargo install --path . --features gpu
```

### Dependencies

You will need the following dependencies:

#### Required
- CUDA Toolkit 12.x or newer (https://developer.nvidia.com/cuda-downloads)
- A compatible NVIDIA GPU (Compute Capability 7.0 or higher)
- Rust compiler (1.76.0 or newer)

#### Linux
```
sudo apt-get install openssl pkg-config libssl-dev
```

#### MacOS
```
brew install openssl pkg-config

# If you encounter issues with OpenSSL, you might need to set the following environment variables:
export PATH="/usr/local/opt/openssl/bin:$PATH"
export LDFLAGS="-L/usr/local/opt/openssl/lib"
export CPPFLAGS="-I/usr/local/opt/openssl/include"
```

#### Windows
```
choco install openssl pkgconfiglite
```

## ⛏️ Collect with GPU

To start collecting with GPU, load your keypair with some ETH, and then use the `collect` command with the `--gpu` flag:

```sh
bitz collect --gpu
```

You can also specify the GPU device to use if you have multiple GPUs:

```sh
bitz collect --gpu --gpu-device 0
```

## 🔧 Optimize Mining

### Set Minimum Difficulty

You can set a custom minimum difficulty to focus on higher-value solutions:

```sh
bitz collect -m 25
```

This will only submit solutions with a difficulty of 25 or higher, which can reduce unnecessary transactions and improve efficiency.

### GPU Optimization Parameters

For fine-tuning GPU performance, you can adjust these parameters:

```sh
bitz collect --gpu --batch-size 768 --hash-threads 320 --solve-threads 160
```

Available parameters:
- `--batch-size`: Size of GPU processing batch (512, 768, 1024, etc.)
- `--hash-threads`: Number of GPU threads per block for hashing (256, 320, 384)
- `--solve-threads`: Number of GPU threads per block for solving (128, 160, 192)

## 🔄 Command Parameters

### 所有参数中文说明

| 参数 | 短参数 | 说明 | 默认值 | 示例 |
| ---- | ------ | ---- | ------ | ---- |
| `--cores` | `-c` | 用于挖矿的CPU核心数，设置为"ALL"使用全部核心 | `1` | `bitz collect -c 4` |
| `--buffer-time` | `-b` | 在截止时间前多少秒停止收集并开始提交 | `5` | `bitz collect -b 10` |
| `--device-id` | `-d` | 用于矿池收集的设备ID（每个密钥最多5个设备） | 无 | `bitz collect -d 2` |
| `--pool-url` | `-p` | 加入并转发解决方案的矿池URL | 无 | `bitz collect -p https://pool.xyz` |
| `--verbose` | `-v` | 是否运行在详细模式 | `false` | `bitz collect -v` |
| `--gpu` | 无 | 使用GPU进行挖矿（如果可用） | `false` | `bitz collect --gpu` |
| `--gpu-device` | 无 | 使用的GPU设备ID | `0` | `bitz collect --gpu --gpu-device 1` |
| `--batch-size` | 无 | GPU批处理大小（512、768、1024等） | `1024` | `bitz collect --gpu --batch-size 768` |
| `--hash-threads` | 无 | 哈希阶段的GPU线程块数量 | `384` | `bitz collect --gpu --hash-threads 320` |
| `--solve-threads` | 无 | 解算阶段的GPU线程块数量 | `192` | `bitz collect --gpu --solve-threads 160` |
| `--min-difficulty` | `-m` | 设置挖矿的最小难度（默认使用链上值） | 链上值 | `bitz collect -m 25` |

### 常用命令组合示例

1. **基本CPU挖矿**
   ```sh
   bitz collect
   ```

2. **使用8个CPU核心挖矿**
   ```sh
   bitz collect -c 8
   ```

3. **使用全部CPU核心和更长的缓冲时间**
   ```sh
   bitz collect -c ALL -b 15
   ```

4. **使用GPU挖矿，最小难度为30**
   ```sh
   bitz collect --gpu -m 30
   ```

5. **详细模式的矿池挖矿**
   ```sh
   bitz collect -p https://pool.example.com -v
   ```

6. **优化配置的GPU挖矿**
   ```sh
   bitz collect --gpu --gpu-device 0 --batch-size 512 --hash-threads 256 --solve-threads 128 -b 10 -m 25
   ```

7. **连接矿池使用GPU挖矿**
   ```sh
   bitz collect --gpu -p https://pool.example.com -d 1
   ```

8.4060实践
   ```
      bitz collect --gpu --batch-size 1024 --hash-threads 384 --solve-threads 192 -m 30
      bitz collect --gpu --batch-size 1024 --hash-threads 256 --solve-threads 128 
      bitz collect --gpu --batch-size 2048 --hash-threads 512 --solve-threads 256 
      ./bitz collect --gpu --batch-size 2048 --hash-threads 1024 --solve-threads 256 
./bitz collect --gpu --batch-size 2048 --hash-threads 256 --solve-threads 128  -c ALL
      bitz collect --gpu --batch-size 2048 --hash-threads 1024 --solve-threads 256 
   ```
## ❓ Help

Add the `-h` flag on any command to pull up a help menu with documentation:

```sh
bitz -h
bitz collect -h
```