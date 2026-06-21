# North-Mini-Code Activation Outlier Analysis

This note summarizes the code added, experiments run, quantitative findings,
and conclusions for activation outlier analysis on
`CohereLabs/North-Mini-Code-1.0`.

## Added Code

### vLLM Runtime Analysis

File: `benchmarks/benchmark_activation_outliers.py`

Purpose:

- Load `CohereLabs/North-Mini-Code-1.0` with vLLM.
- Apply ShareGPT-style chat prompts using the tokenizer chat template.
- Register runtime hooks through `LLM.apply_model()`.
- Capture hook-visible input activations for:
    - attention `q/k/v` shared input via vLLM fused `qkv_proj`
    - attention `o_proj` input
    - dense MLP `gate_up_proj` input
    - dense MLP `down_proj` input
    - sparse MoE router input
    - sparse MoE expert input
- Simulate activation quantization error for:
    - `nvfp4` with global scale fixed to `1`
    - `mxfp4`
    - `mxfp8`
    - `fp8_ptpc`
    - `block_fp8`, group size `128`
- Generate local HTML/JSON/CSV reports.
- Put MoE activation heat graphs first.
- Add token/channel/magnitude plots to inspect fixed-channel outliers.

Important implementation detail:

- This script sets `VLLM_ALLOW_INSECURE_SERIALIZATION=1` because this vLLM
  branch requires pickle/cloudpickle serialization for callable
  `LLM.apply_model()` RPCs.
- vLLM exposes the sparse MoE block as a fused `FusedMoE`, so routed expert
  internal `down_proj` inputs are not directly visible from normal module hooks.

### Hugging Face Transformers Verification

File: `benchmarks/benchmark_activation_outliers_hf.py`

Purpose:

- Load the same model with Hugging Face Transformers.
- Use the same chat-template-rendered prompts.
- Hook matching activation inputs:
    - attention `q_proj/k_proj/v_proj` input
    - attention `o_proj` input
    - dense MLP `gate_proj/up_proj/down_proj` input
    - sparse MoE router input
    - sparse MoE expert input
- Monkey-patch `Cohere2MoeExperts.forward()` to capture exact routed expert
  internals:
    - `expert_gate_up_input`
    - `expert_down_input`, i.e. `silu(gate) * up`, the input to expert
    `down_proj`
- Compare HF hook-visible activation stats with the vLLM JSON report.

### Targeted Follow-Up Analyses

Generated local analysis files:

- `north_activation_outliers_hf_smoke/expert_down_channel_full_summary.json`
- `north_activation_outliers_hf_smoke/expert_down_same_expert_token_channel_overlap.json`
- `north_activation_outliers_hf_smoke/fixed_module_channel_stability.json`

These files were produced by targeted HF passes to answer specific follow-up
questions about channel recurrence, token overlap, and fixed-module stability.

## Experiments Run

### vLLM Smoke / Main Report

Command:

```bash
.venv/bin/python benchmarks/benchmark_activation_outliers.py \
  --model CohereLabs/North-Mini-Code-1.0 \
  --max-prompts 1 \
  --max-tokens 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.6 \
  --max-rows-per-hook 64 \
  --max-plot-tokens 32 \
  --max-plot-channels 48 \
  --num-moe-3d-plots 2 \
  --quant-schemes nvfp4 mxfp4 mxfp8 fp8_ptpc block_fp8 \
  --block-fp8-group-size 128 \
  --output-dir north_activation_outliers_chat_smoke
```

Outputs:

- `north_activation_outliers_chat_smoke/activation_outliers.html`
- `north_activation_outliers_chat_smoke/activation_outliers.json`
- `north_activation_outliers_chat_smoke/activation_outliers.csv`

Result:

- Installed `196` hooks.
- This matches expected coverage:
    - 49 attention qkv inputs
    - 49 attention o inputs
    - 1 dense MLP gate/up input
    - 1 dense MLP down input
    - 48 sparse MoE router inputs
    - 48 sparse MoE expert inputs

### HF Verification

Command:

```bash
.venv/bin/python benchmarks/benchmark_activation_outliers_hf.py \
  --model CohereLabs/North-Mini-Code-1.0 \
  --max-prompts 1 \
  --max-tokens 1 \
  --max-rows-per-hook 64 \
  --max-plot-tokens 32 \
  --max-plot-channels 48 \
  --num-moe-3d-plots 2 \
  --quant-schemes nvfp4 mxfp4 mxfp8 fp8_ptpc block_fp8 \
  --block-fp8-group-size 128 \
  --compare-vllm-json north_activation_outliers_chat_smoke/activation_outliers.json \
  --output-dir north_activation_outliers_hf_smoke
```

Outputs:

- `north_activation_outliers_hf_smoke/activation_outliers_hf.html`
- `north_activation_outliers_hf_smoke/activation_outliers_hf.json`
- `north_activation_outliers_hf_smoke/activation_outliers_hf.csv`
- `north_activation_outliers_hf_smoke/hf_vllm_comparison.json`
- `north_activation_outliers_hf_smoke/hf_vllm_comparison.csv`

Result:

- Installed `196` standard HF hooks.
- Patched `48` sparse expert modules.
- HF/vLLM shared activation max-abs relative delta:
    - mean: `0.865%`
    - max: `8.99%`

Conclusion:

- HF and vLLM hook-visible activation results are very close for shared hook
  points, validating the vLLM activation capture path.

### Validation

Commands:

```bash
.venv/bin/python -m py_compile \
  benchmarks/benchmark_activation_outliers.py \
  benchmarks/benchmark_activation_outliers_hf.py

pre-commit run ruff-check --files \
  benchmarks/benchmark_activation_outliers.py \
  benchmarks/benchmark_activation_outliers_hf.py

pre-commit run ruff-format --files \
  benchmarks/benchmark_activation_outliers.py \
  benchmarks/benchmark_activation_outliers_hf.py
```

Result:

- `py_compile`: passed
- `ruff-check`: passed
- `ruff-format`: passed

## Results

## 1. MoE Boundary Hidden-State Outlier Channels

This is the input to sparse MoE router / expert at hidden size `2048`.

In vLLM, top recurring MoE boundary channels were:

| channel | layers in top sampled channels | max abs | mean layer max | sampled values > 4 |
|---:|---:|---:|---:|---:|
| 1151 | 48 / 48 | 9.5625 | 4.6673 | 60 |
| 1943 | 48 / 48 | 6.5313 | 3.8245 | 28 |
| 1474 | 48 / 48 | 6.5000 | 5.1471 | 39 |
| 1920 | 47 / 48 | 8.0625 | 5.8137 | 38 |
| 1730 | 45 / 48 | 7.5313 | 5.1720 | 284 |
| 239 | 42 / 48 | 7.0000 | 5.0030 | 35 |
| 168 | 41 / 48 | 7.6563 | 4.3904 | 54 |

HF matched closely:

| channel | layers in top sampled channels | max abs | mean layer max | sampled values > 4 |
|---:|---:|---:|---:|---:|
| 1151 | 48 / 48 | 9.5000 | 4.6419 | 60 |
| 1943 | 48 / 48 | 6.5625 | 3.8473 | 28 |
| 1474 | 48 / 48 | 6.3750 | 5.1064 | 39 |
| 1920 | 47 / 48 | 8.0625 | 5.7395 | 38 |
| 1730 | 45 / 48 | 7.5000 | 5.1632 | 273 |
| 239 | 42 / 48 | 7.1875 | 5.2000 | 35 |
| 168 | 41 / 48 | 7.7188 | 4.4045 | 54 |

Conclusion:

- MoE boundary hidden-state outliers are clearly concentrated in recurring
  hidden channels.
- Channels such as `1151`, `1943`, `1474`, `1920`, `1730`, `239`, and `168`
  recur across many sparse MoE layers.
- Channel `1730` is especially persistent across tokens, while channel `1151`
  has the largest peak and appears in every sparse MoE layer.

## 2. Expert Gate/Up Input Outliers

This is the input to each routed expert's fused gate/up projection. It is still
in hidden-size space (`2048`).

Top exact HF expert gate/up input layer maxima:

| layer | max abs | p999 abs | outlier ratio |
|---:|---:|---:|---:|
| 45 | 10.9375 | 5.9063 | 5.679 |
| 47 | 9.5000 | 3.2254 | 5.344 |
| 48 | 9.4375 | 6.2500 | 6.787 |
| 44 | 8.2500 | 5.5313 | 4.567 |
| 28 | 8.0625 | 6.2015 | 5.864 |
| 43 | 8.0000 | 5.5703 | 4.559 |
| 46 | 8.0000 | 5.7500 | 4.274 |

Conclusion:

- Expert gate/up input has outliers at roughly the same scale as the MoE
  boundary hidden-state input.
- This is expected because it is the same hidden-channel space.

## 3. Expert Down-Projection Input Outliers

This is the exact input to expert `down_proj`, i.e. `silu(gate) * up`.
It lives in expert intermediate-size space (`768`), not hidden-size space.

Top layer-level exact expert down input maxima:

| layer | max abs | p999 abs | outlier ratio |
|---:|---:|---:|---:|
| 48 | 1608.0 | 749.939 | 425.326 |
| 24 | 1152.0 | 252.750 | 881.650 |
| 45 | 398.0 | 182.289 | 118.850 |
| 25 | 80.0 | 30.486 | 64.309 |
| 11 | 71.0 | 5.445 | 112.198 |
| 46 | 63.5 | 10.401 | 15.482 |
| 1 | 56.0 | 1.214 | 183.795 |

Full per-channel pass for expert down input:

| intermediate channel | max abs | values > 512 | values > 128 | values > 64 | values > 16 |
|---:|---:|---:|---:|---:|---:|
| 318 | 1608 | 2 | 2 | 2 | 2 |
| 628 | 1152 | 2 | 2 | 2 | 2 |
| 311 | 414 | 0 | 2 | 2 | 3 |
| 123 | 398 | 0 | 2 | 2 | 2 |
| 540 | 290 | 0 | 2 | 2 | 2 |
| 748 | 258 | 0 | 2 | 2 | 2 |
| 608 | 244 | 0 | 2 | 2 | 2 |
| 287 | 212 | 0 | 2 | 2 | 3 |
| 519 | 210 | 0 | 2 | 2 | 2 |
| 337 | 205 | 0 | 2 | 2 | 2 |

Top exact locations:

| layer | expert | channel | abs | token idx | top-k pos |
|---:|---:|---:|---:|---:|---:|
| 48 | 23 | 318 | 1608 | 0 | 0 |
| 24 | 22 | 628 | 1152 | 99 | 0 |
| 45 | 87 | 123 | 398 | 109 | 0 |
| 48 | 22 | 690 | 201 | 0 | 4 |
| 48 | 38 | 746 | 188 | 0 | 7 |

Conclusion:

- Expert `down_proj` input outliers are much larger than MoE boundary outliers.
- These extreme events are sparse and usually tied to a specific
  `(layer, expert, token, channel)` combination.
- The largest channels `318`, `628`, and `123` dominate peak magnitude, but they
  are not globally fixed across all layers.

## 4. Are Expert Down Outlier Channels Shared Across Tokens?

For `abs >= 16`, token-level expert down outliers were:

| token idx | events | unique channels | max abs |
|---:|---:|---:|---:|
| 0 | 12 | 12 | 1608 |
| 2 | 12 | 12 | 80 |
| 109 | 8 | 8 | 398 |
| 99 | 4 | 4 | 1152 |
| 4 | 3 | 3 | 38.5 |
| 125 | 3 | 3 | 19.625 |

Pairwise token overlap among the most active tokens:

- token `0` vs token `2`: no overlap
- token `0` vs token `109`: no overlap
- token `0` vs token `99`: no overlap
- token `2` vs token `109`: only channel `679` overlaps
- token `99` vs top others: no overlap

Only four channels appeared in more than one token among `abs >= 16` events:

| channel | tokens |
|---:|---|
| 753 | `89, 190` |
| 679 | `2, 109` |
| 402 | `99, 115` |
| 345 | `49, 132` |

Conclusion:

- Across arbitrary tokens, expert down input outlier channels are generally not
  the same.
- They are token-specific and expert-specific.

## 5. Fixed `(Layer, Expert)` Token Overlap

When fixing the exact expert module as `(layer, expert_id)`, and considering
only `expert_down_input abs >= 16`:

- extreme channel events: `211`
- `(layer, expert)` groups with two or more outlier tokens: `30`
- token pairs within fixed `(layer, expert)`: `116`
- token pairs with any channel overlap: `79 / 116 = 68.1%`
- mean Jaccard overlap: `0.634`

Examples:

| fixed module | tokens | channel overlap |
|---|---|---|
| `L45 expert_87` | `99,109` | identical channels |
| `L24 expert_22` | `99,109` | Jaccard `0.846` |
| `L44 expert_127` | many tokens | many hit `513`, some hit `409` or `395` |
| `L45 expert_26` | `80,107,164` | shared channel `15` |
| `L46 expert_116` | `67,42,91` | no overlap |

Conclusion:

- If we fix the exact `(layer, expert)`, multiple tokens with extreme expert
  down outliers often share outlier channels.
- However, this is not guaranteed; some fixed experts show token-specific
  outlier channels.

## 6. Fixed `(Layer, Module)` Stability Across Module Types

Using top-8 absolute channels per token and measuring token-pair Jaccard:

| module type | modules | mean Jaccard top-8 | median Jaccard | top1 mode fraction |
|---|---:|---:|---:|---:|
| attention qkv input | 49 | 0.066 | 0.061 | 0.343 |
| attention o input | 49 | 0.173 | 0.137 | 0.233 |
| expert gate/up input | 3953 | 0.125 | 0.089 | 0.513 |
| expert gate/up input, modules with >=4 tokens | 3007 | 0.099 | 0.088 | 0.462 |
| expert down input | 3953 | 0.067 | 0.030 | 0.273 |
| expert down input, modules with >=4 tokens | 3007 | 0.042 | 0.030 | 0.196 |

For `expert_down_input abs >= 16`, fixed `(layer, expert)` groups had:

- groups: `30`
- average pair overlap fraction: `0.789`
- average threshold-channel Jaccard: `0.770`

Conclusion:

- Ordinary top-k channels are not very stable for most fixed modules.
- True extreme expert down outliers are much more stable after fixing
  `(layer, expert)`.
- Attention qkv inputs have weak top-k stability, even though certain hidden
  channels recur globally across layers.
- Attention o input is somewhat more stable than qkv input in some layers.

## Overall Conclusions

1. vLLM activation capture is validated.
   HF and vLLM shared hook results are close: mean max-abs relative delta
   `0.865%`, max `8.99%`.

2. MoE boundary hidden-state outliers are fixed-channel-like.
   Channels such as `1151`, `1943`, `1474`, `1920`, `1730`, `239`, and `168`
   recur across many sparse MoE layers.

3. Expert gate/up input has similar outlier scale to MoE boundary hidden input.
   This is expected because it is still hidden-size input.

4. Expert down input is the most extreme activation site.
   It can reach magnitudes above `1000`, e.g. layer 48 expert 23 channel 318
   reaches `1608`.

5. Expert down input outliers are sparse and module-specific.
   Across arbitrary tokens, channels differ. After fixing `(layer, expert)`,
   extreme outlier channels often overlap, but not always.

6. For quantization risk:
   - MoE boundary input suggests fixed hidden-channel outlier handling may help.
   - Expert down input suggests per-expert or per-token adaptive handling is
     needed, because the largest spikes are tied to specific routed expert
     paths and tokens.

## Current Artifacts

Scripts:

- `benchmarks/benchmark_activation_outliers.py`
- `benchmarks/benchmark_activation_outliers_hf.py`

vLLM output:

- `north_activation_outliers_chat_smoke/activation_outliers.html`
- `north_activation_outliers_chat_smoke/activation_outliers.json`
- `north_activation_outliers_chat_smoke/activation_outliers.csv`

HF output:

- `north_activation_outliers_hf_smoke/activation_outliers_hf.html`
- `north_activation_outliers_hf_smoke/activation_outliers_hf.json`
- `north_activation_outliers_hf_smoke/activation_outliers_hf.csv`
- `north_activation_outliers_hf_smoke/hf_vllm_comparison.json`
- `north_activation_outliers_hf_smoke/hf_vllm_comparison.csv`

Targeted analysis output:

- `north_activation_outliers_hf_smoke/expert_down_channel_full_summary.json`
- `north_activation_outliers_hf_smoke/expert_down_same_expert_token_channel_overlap.json`
- `north_activation_outliers_hf_smoke/fixed_module_channel_stability.json`
