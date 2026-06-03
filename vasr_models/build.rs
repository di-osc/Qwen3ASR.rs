fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=kernels/mmvq_gguf/mmvq_gguf.cu");

        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let out_file = out_dir.join("libqwen3_asr_q8_mmvq.a");
        bindgen_cuda::Builder::default()
            .kernel_paths(vec!["kernels/mmvq_gguf/mmvq_gguf.cu"])
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--use_fast_math")
            .arg("--compiler-options")
            .arg("-fPIC")
            .build_lib(&out_file);

        println!("cargo:rustc-link-search={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=qwen3_asr_q8_mmvq");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
