# Qwen3-ASR Paged Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Metal-only paged attention decode path inspired by `xinfer`, while keeping the current dynamic KV cache path as the default fallback.

**Architecture:** Introduce an optional `metal-paged-attn` feature that wires `attention-rs` into the text decoder. The paged path owns preallocated K/V cache pages, builds `InputMetadata` style slot/block tensors for batch size 1 dense prompts, and bypasses per-token K/V `Tensor::cat` during decode.

**Tech Stack:** Rust, Candle 0.9, optional `attention-rs`, Metal, PyO3/maturin for end-to-end verification.

---

## File Structure

- Modify `qwen3_asr_runtime/Cargo.toml`: add optional `attention-rs` dependency and feature gates.
- Create `qwen3_asr_runtime/src/model/paged_kv_cache.rs`: preallocated page tensors, block table, slot mapping, and metadata helpers.
- Modify `qwen3_asr_runtime/src/model/mod.rs`: expose `paged_kv_cache` behind the feature.
- Modify `qwen3_asr_runtime/src/model/thinker_text.rs`: create paged attention objects per layer and add paged forward methods.
- Modify `qwen3_asr_runtime/src/model/thinker.rs`: expose a thinker-level paged forward method.
- Modify `qwen3_asr_runtime/src/model/generation.rs`: add an opt-in paged greedy path for batch size 1 dense prompts.
- Modify `qwen3_asr_runtime/src/inference/transcribe.rs`: choose paged generation only when feature/device/input constraints match.
- Test with `cargo test`, `cargo test --features metal-paged-attn`, `maturin develop`, and the fixture audio.

---

### Task 1: Dependency Probe

**Files:**
- Modify: `qwen3_asr_runtime/Cargo.toml`

- [ ] **Step 1: Add optional dependency and feature gates**

Add this dependency:

```toml
attention-rs = { git = "https://github.com/guoqingbao/attention.rs.git", rev = "8fdb7ab", optional = true }
```

Add these features:

```toml
paged-attn = ["dep:attention-rs"]
metal-paged-attn = ["metal", "paged-attn", "attention-rs/metal", "attention-rs/metal-flash"]
```

- [ ] **Step 2: Compile the feature probe**

Run:

```bash
cargo check -p qwen3_asr_runtime --features metal-paged-attn
```

Expected success: Cargo resolves one Candle version and the crate compiles far enough to reach this project code.

Expected dependency failure: duplicate `candle-core` type errors or git dependency conflicts. If this happens, stop implementation and record the exact conflict; the next design decision is whether to move this repo to the `xinfer` Candle fork or vendor an `attention-rs` adapter.

- [ ] **Step 3: Commit**

```bash
git add qwen3_asr_runtime/Cargo.toml Cargo.lock
git commit -m "build: add optional paged attention dependency"
```

---

### Task 2: Paged KV Cache Unit

**Files:**
- Create: `qwen3_asr_runtime/src/model/paged_kv_cache.rs`
- Modify: `qwen3_asr_runtime/src/model/mod.rs`

- [ ] **Step 1: Write tests for block math**

Add tests that verify:

```rust
#[test]
fn test_paged_cache_single_sequence_layout() -> anyhow::Result<()> {
    let device = candle_core::Device::Cpu;
    let cache = PagedKvCache::new(
        2,
        4,
        8,
        16,
        40,
        candle_core::DType::F32,
        &device,
    )?;
    assert_eq!(cache.block_size(), 16);
    assert_eq!(cache.num_blocks(), 3);
    assert_eq!(cache.block_table_host(), &[0, 1, 2]);
    assert_eq!(cache.slot_for_position(0)?, 0);
    assert_eq!(cache.slot_for_position(16)?, 16);
    assert_eq!(cache.slot_for_position(39)?, 39);
    Ok(())
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cargo test -p qwen3_asr_runtime model::paged_kv_cache::tests::test_paged_cache_single_sequence_layout
```

Expected: FAIL because `PagedKvCache` does not exist.

- [ ] **Step 3: Implement `PagedKvCache`**

Create a struct with:

```rust
pub struct PagedKvCache {
    key_cache: Vec<candle_core::Tensor>,
    value_cache: Vec<candle_core::Tensor>,
    block_table_host: Vec<i32>,
    block_size: usize,
    num_blocks: usize,
    max_tokens: usize,
}
```

Allocate per-layer tensors with shapes expected by `attention-rs`:

```rust
// key cache: (num_blocks, num_kv_heads, head_dim / x, block_size, x)
// value cache: (num_blocks, num_kv_heads, head_dim, block_size)
```

Use `x = 16 / dtype.size_in_bytes()` for the key cache layout. Validate that `head_dim % x == 0`.

- [ ] **Step 4: Add metadata helpers**

Implement:

```rust
pub fn slot_mapping_for_range(&self, start: usize, len: usize, device: &Device) -> Result<Tensor>;
pub fn block_tables_tensor(&self, device: &Device) -> Result<Tensor>;
pub fn context_lens_tensor(&self, context_len: usize, device: &Device) -> Result<Tensor>;
pub fn key_value_cache(&self, layer_idx: usize) -> Result<(&Tensor, &Tensor)>;
```

For phase 1, `block_tables_tensor` returns shape `(1, num_blocks)` and `context_lens_tensor` returns shape `(1,)`.

- [ ] **Step 5: Run tests**

Run:

```bash
cargo test -p qwen3_asr_runtime model::paged_kv_cache::tests
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add qwen3_asr_runtime/src/model/paged_kv_cache.rs qwen3_asr_runtime/src/model/mod.rs
git commit -m "feat: add paged kv cache metadata"
```

---

### Task 3: Text Attention Paged Forward

**Files:**
- Modify: `qwen3_asr_runtime/src/model/thinker_text.rs`

- [ ] **Step 1: Add feature-gated imports and layer state**

Behind `#[cfg(feature = "paged-attn")]`, import:

```rust
use attention_rs::{InputMetadata, PagedAttention};
```

Add to `ThinkerTextAttention`:

```rust
#[cfg(feature = "paged-attn")]
paged_attn: Option<PagedAttention>,
```

Initialize it in `load` with:

```rust
PagedAttention::new(
    num_attention_heads,
    head_dim,
    (head_dim as f32).powf(-0.5),
    Some(num_key_value_heads),
    None,
    device.clone(),
    None,
    false,
)?
```

If the exact constructor differs during compilation, adjust only this constructor call to match the checked-out `attention-rs` API.

- [ ] **Step 2: Implement layer paged forward**

Add `forward_with_paged_cache` that reuses the existing Q/K/V projection, Q/K norm, and mRoPE application. Pass Q/K/V to:

```rust
paged_attn.forward(
    &q,
    &k,
    &v,
    None,
    Some(key_cache.clone()),
    Some(value_cache.clone()),
    input_metadata,
    None,
)
```

Reshape the returned attention output back to `(batch, seq_len, num_attention_heads * head_dim)` and apply `o_proj`.

- [ ] **Step 3: Keep old path untouched**

Do not alter `forward_with_kv_cache` except for sharing private helper code if the compiler requires it. Existing tests must continue to pass without `paged-attn`.

- [ ] **Step 4: Compile both feature sets**

Run:

```bash
cargo check -p qwen3_asr_runtime
cargo check -p qwen3_asr_runtime --features metal-paged-attn
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add qwen3_asr_runtime/src/model/thinker_text.rs
git commit -m "feat: add paged text attention forward path"
```

---

### Task 4: Paged Greedy Generation

**Files:**
- Modify: `qwen3_asr_runtime/src/model/thinker.rs`
- Modify: `qwen3_asr_runtime/src/model/generation.rs`

- [ ] **Step 1: Add a failing dense-path test**

Add a unit test for a helper:

```rust
#[test]
fn test_paged_generation_eligibility() {
    assert!(can_use_paged_decode(1, &[1, 1, 1]));
    assert!(!can_use_paged_decode(2, &[1, 1, 1]));
    assert!(!can_use_paged_decode(1, &[0, 1, 1]));
}
```

- [ ] **Step 2: Implement eligibility**

Add:

```rust
fn can_use_paged_decode(batch: usize, attention_mask: &[u32]) -> bool {
    batch == 1 && attention_mask.iter().all(|&v| v != 0)
}
```

- [ ] **Step 3: Add `greedy_generate_paged`**

Implement it behind `#[cfg(feature = "paged-attn")]`. It should mirror `greedy_generate_cached` but:

- allocates `PagedKvCache` for `input_ids.len() + max_new_tokens`,
- uses slot mapping `0..seq_len` for prefill,
- uses a one-token slot mapping for each decode step,
- builds `InputMetadata` with `is_prefill`, `slot_mapping`, `block_tables`, `context_lens`, and max sequence lengths,
- calls the new thinker paged forward method.

- [ ] **Step 4: Add fallback selection**

Add a wrapper:

```rust
pub fn greedy_generate_auto(...) -> Result<Vec<u32>>
```

It calls paged generation only when the feature is enabled, the device is Metal, `batch == 1`, and the mask is dense. Otherwise it calls `greedy_generate_cached`.

- [ ] **Step 5: Run tests**

Run:

```bash
cargo test -p qwen3_asr_runtime model::generation::tests
cargo check -p qwen3_asr_runtime --features metal-paged-attn
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add qwen3_asr_runtime/src/model/thinker.rs qwen3_asr_runtime/src/model/generation.rs
git commit -m "feat: add paged greedy decode path"
```

---

### Task 5: Inference Wiring and Verification

**Files:**
- Modify: `qwen3_asr_runtime/src/inference/transcribe.rs`

- [ ] **Step 1: Route single-item transcription through auto generation**

Replace the direct call to `greedy_generate_cached_batch` only for the single-row dense path. Keep batch transcription on the current batch function.

- [ ] **Step 2: Build Metal wheel**

Run:

```bash
maturin develop --features metal,metal-paged-attn
```

Expected: build succeeds and installs the local extension.

- [ ] **Step 3: Verify fixture text**

Run:

```bash
python - <<'PY'
from qwen3_asr_rs import Qwen3ASR
model = Qwen3ASR.from_pretrained("Qwen/Qwen3-ASR-0.6B", device="metal")
res = model.transcribe("fixtures/audio/asr_en_16k.wav", language="English")
print(res.text)
PY
```

Expected text:

```text
Hmm. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.
```

- [ ] **Step 4: Compare speed**

Run the same script twice with and without `metal-paged-attn`, timing only `model.transcribe(...)`. Report both averages.

- [ ] **Step 5: Full verification**

Run:

```bash
cargo fmt --all -- --check
cargo test --workspace
python -m pytest tests/test_python_api.py -q
cargo test -p qwen3_asr_runtime --features metal-paged-attn
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add qwen3_asr_runtime/src/inference/transcribe.rs
git commit -m "feat: enable paged attention transcription path"
```

---

## Self-Review

Spec coverage:

- Metal-only paged attention: Task 1, Task 3, Task 5.
- Preallocated KV cache: Task 2.
- Metadata-driven decode: Task 2 and Task 4.
- Existing fallback behavior: Task 3 and Task 4.
- ASR API preservation: Task 5.
- Verification and speed comparison: Task 5.

Placeholder scan:

- The plan includes one explicit compatibility branch in Task 1 because dependency compatibility cannot be known without compiling. It has a concrete stop condition and reporting requirement.

Type consistency:

- `PagedKvCache`, `InputMetadata`, `PagedAttention`, `greedy_generate_paged`, and `can_use_paged_decode` are introduced before use.
