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

## ❓ Help

Add the `-h` flag on any command to pull up a help menu with documentation:

```sh
bitz -h
bitz collect -h
```