#[cfg(not(feature = "gpu"))]
fn main() {}

#[cfg(feature = "gpu")]
fn main() {
    println!("cargo:rerun-if-changed=cuda/");
    println!("cargo:rerun-if-changed=src/");

    // 检查CUDA目录
    let mut cuda_ok = true;
    
    for path in &[
        "cuda/equix/include",
        "cuda/equix/src",
        "cuda/hashx/include",
        "cuda/hashx/src",
    ] {
        if !std::path::Path::new(path).exists() {
            println!("cargo:warning=目录不存在: {}", path);
            cuda_ok = false;
        }
    }

    if !cuda_ok {
        panic!("CUDA目录结构不完整，请确保所有必要的CUDA库文件都存在");
    }

    // 编译bitz_drillx
    cc::Build::new()
        .cuda(true)
        .include("cuda/equix/include")
        .include("cuda/equix/src")
        .include("cuda/hashx/include")
        .include("cuda/hashx/src")
        .file("cuda/bitz_drillx.cu")
        .file("cuda/equix/src/context.cu")
        .file("cuda/equix/src/equix.cu")
        .file("cuda/equix/src/solver.cu")
        .file("cuda/hashx/src/blake2.cu")
        .file("cuda/hashx/src/compiler.cu")
        .file("cuda/hashx/src/context.cu")
        .file("cuda/hashx/src/hashx.cu")
        .file("cuda/hashx/src/program.cu")
        .file("cuda/hashx/src/program_exec.cu")
        .file("cuda/hashx/src/siphash.cu")
        .file("cuda/hashx/src/siphash_rng.cu")
        .flag("-cudart=static")
        .flag("-diag-suppress=174")
        // 高级优化标志
        .flag("-O3")  // 最高级别优化
        .flag("-use_fast_math")  // 使用快速数学库
        .flag("-Xptxas=-v")  // 在编译时显示寄存器和内存使用情况
        .flag("-maxrregcount=64")  // 限制每线程的寄存器使用，提高并行度
        // 支持多种GPU架构，但针对每种架构进行特殊优化
        // .flag("-gencode=arch=compute_70,code=sm_70") // Volta (V100)
        // .flag("-gencode=arch=compute_75,code=sm_75") // Turing (RTX 2080)
        // .flag("-gencode=arch=compute_80,code=sm_80") // Ampere (A100, RTX 3080)
        // .flag("-gencode=arch=compute_86,code=[sm_86,compute_86]") // Ampere (RTX 3060, 3070, 3080 Ti) - 主要目标
        .flag("-gencode=arch=compute_89,code=[sm_89,compute_89]") // Ada Lovelace (RTX 4090, etc) - 主要目标
        .compile("bitz_drillx.a");

    // 添加链接库目录 - 支持更多系统
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\lib\\x64");
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    // 输出编译库的位置
    let out_dir = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-search=native={}", out_dir);
} 