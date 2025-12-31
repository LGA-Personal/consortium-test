# Known Issues and Limitations

This document tracks known issues, bugs, and limitations discovered during development that require future fixes or workarounds.

---

## 🔴 Critical Issues

### MoE Model Sharding Incompatibility

**Status:** ❌ Broken  
**Affected Models:** Mixtral-8x7B, DeepSeek-MoE, and other Mixture-of-Experts architectures  
**Severity:** High  
**Discovered:** 2025-12-30

#### Description

Distributed sharding fails for Mixture-of-Experts (MoE) models due to incompatibility between MLX's `model.sanitize()` function and partial layer loading.

#### Error

```
KeyError: 'model.layers.19.block_sparse_moe.experts.5.w1.weight'
```

#### Root Cause

1. `mlx_lm.utils.load_model()` calls `model.sanitize(weights)` which expects ALL expert weights to be present
2. With pipeline sharding, each node only loads a subset of layers (e.g., layers 0-15 on Node A, 16-31 on Node B)
3. The `sanitize()` function in `mlx_lm/models/mixtral.py` attempts to pop expert weights for layers that don't exist in the current shard
4. This causes a `KeyError` and crashes the runner process

#### Affected Code

- `src/exo/worker/engines/mlx/utils_mlx.py::shard_and_load()` (line 263)
- `mlx_lm/models/mixtral.py::sanitize()` (external dependency)

#### Workaround

Use non-MoE models for distributed inference:

- ✅ Llama-3.x series
- ✅ Qwen2.5 series
- ✅ Mistral (non-MoE variants)
- ❌ Mixtral series
- ❌ DeepSeek-MoE series

#### Potential Fix

Modify `shard_and_load()` to handle MoE models specially:

1. Load the full model structure first
2. Apply `sanitize()` before sharding
3. Then extract only the required layers for this shard

OR

2. Skip `sanitize()` for sharded loads and handle weight cleanup manually
3. Ensure expert weights are only accessed for layers present in the shard

#### References

- Error log: Machine B terminal, 2025-12-30 19:24:21
- Discussion: This conversation, Step 603

---

## ⚠️ Medium Priority Issues

### Network Topology Flapping

**Status:** ✅ Fixed (2025-12-30)  
**Severity:** Medium

#### Description

Reachability checks were too aggressive (1s timeout), causing false-positive connection drops under load.

#### Fix Applied

Increased timeout in `src/exo/worker/utils/net_profile.py` from 1s to 5s.

---

### Dashboard Assets Required for API

**Status:** ✅ Fixed (2025-12-30)  
**Severity:** Low

#### Description

API server required built dashboard assets even when running headless worker nodes.

#### Fix Applied

Made dashboard mounting optional in `src/exo/master/api.py` with graceful fallback.

---

## 📝 Future Enhancements

### Memory Check Bypass

**Status:** ⚠️ Temporary Workaround  
**Severity:** Low

#### Description

Strict memory availability check was preventing model placement even when OS could handle swapping/cache eviction.

#### Current State

Memory check is bypassed in `src/exo/master/placement.py` (lines 62-66, commented out).

#### Future Work

Implement smarter memory estimation that accounts for:

- MLX lazy loading
- OS memory management
- Swap availability
- Per-shard memory requirements

---

## 🔧 How to Add New Issues

When you discover a new issue:

1. **Determine severity:**

   - 🔴 **Critical:** Blocks core functionality
   - ⚠️ **Medium:** Workaround exists but impacts UX
   - 📝 **Enhancement:** Nice-to-have improvement

2. **Add to appropriate section** with:

   - **Status:** ❌ Broken / ⚠️ Workaround / ✅ Fixed
   - **Affected components/models**
   - **Severity level**
   - **Discovery date**
   - **Description** of the issue
   - **Error message** (if applicable)
   - **Root cause** analysis
   - **Affected code** locations
   - **Workaround** (if available)
   - **Potential fix** approach
   - **References** to logs/discussions

3. **Update status** when:
   - A workaround is found (❌ → ⚠️)
   - Issue is fixed (⚠️ → ✅)
   - Move to appropriate section if severity changes
