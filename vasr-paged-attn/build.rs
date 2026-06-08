#[cfg(feature = "metal")]
fn main() -> Result<(), String> {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/metal/kernels/prefill_paged_attn.metal");
    println!("cargo:rerun-if-env-changed=VASR_METAL_PRECOMPILE");

    let skip = env::var("VASR_METAL_PRECOMPILE")
        .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let out_dir = PathBuf::from(env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
    let lib_path = out_dir.join("vasr_varlen_prefill.metallib");

    if skip {
        std::fs::write(&lib_path, []).map_err(|e| e.to_string())?;
        return Ok(());
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").map_err(|_| "CARGO_MANIFEST_DIR")?);
    let source = manifest_dir.join("src/metal/kernels/prefill_paged_attn.metal");
    let working_directory = out_dir.to_string_lossy().to_string();
    let air_path = out_dir.join("prefill_paged_attn.air");

    let mut compile_air = Command::new("xcrun");
    compile_air
        .arg("--sdk")
        .arg("macosx")
        .arg("metal")
        .arg("-std=metal3.1")
        .arg(format!("-working-directory={working_directory}"))
        .arg("-O3")
        .arg("-c")
        .arg("-w")
        .arg(&source);
    let status = compile_air.status().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!(
            "compiling varlen prefill metal -> air failed: {status}"
        ));
    }

    let mut compile_lib = Command::new("xcrun");
    compile_lib
        .arg("metal")
        .arg("-o")
        .arg(&lib_path)
        .arg(&air_path);
    let status = compile_lib.status().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!(
            "compiling varlen prefill air -> metallib failed: {status}"
        ));
    }
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn main() {}
