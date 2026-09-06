# Structured Information Extraction Benchmark: GLiNER2.5 vs UIE (16GB Baseline)

[![Structured IE CI](https://github.com/Ira-Austen/structured-ie-eval/actions/workflows/eval.yml/badge.svg)](https://github.com/Ira-Austen/structured-ie-eval/actions/workflows/eval.yml)

A reproducible, production-oriented end-to-end benchmark comparing **GLiNER2.5-multi (FP32)** and **UIE (UIE-medium & UIE-mini)** for structured information extraction under a **16GB physical memory baseline** (targeting domestic 信创 / UOS production deployments).

---

## 1. Background & Objectives

Earlier exploratory trials on resource-constrained development containers (512–650 MiB) previously raised concerns regarding the memory footprint of GLiNER2.5. Following confirmation that target 信创 production hosts feature **16GB RAM**, this repository establishes a definitive, unconstrained comparison:

1. **Quantify GLiNER2.5 Feature Gains**: Quantifies the empirical gains from GLiNER2.5's core architectural innovations:
   - **JointIE (G2)**: Joint candidate lattice scoring with graph constraints (acyclic, no self-loops, endpoint type validation).
   - **Natural Records (G3)**: Anchor-based instance grouping and cardinality constraints to eliminate field cross-pairing in multi-event texts.
   - **Mention Attributes & Constrained Classification (G4a, G4b)**: Precise binding of modality (actual vs intended vs conditional vs negated) directly to event mentions.
   - **Integrated Pipeline (G5)**: Full pipeline combining JointIE relations, natural records, and span attributes.
2. **Fair Cross-Model Evaluation**: Evaluates GLiNER against `Casually/uie-medium` (75.4M params) and `Casually/uie-mini` (27.0M params) using:
   - Strict character-level tokenization (`char` word splitter for Chinese).
   - Explicit `batch_size=1` for fair single-stream comparison.
   - Canonical directed fact contracts and relation direction normalization.
   - Identical external validation rules (G1) applied across all models.
3. **Hardware & Resource Profiling**: Continuous cgroup v2 and process telemetry sampling (0.5s intervals) measuring:
   - True physical memory peaks (RSS and kernel memory peak).
   - Zero-Swap execution guarantee.
   - 1, 2, and 4 thread scaling curves.
   - Long-text sliding window streaming (1k, 4k, 16k characters).

---

## 2. Repository Structure

```
structured-ie-eval/
├── .github/
│   └── workflows/
│       └── eval.yml                  # One-click workflow_dispatch for full matrix run
├── config/
│   ├── manifest.json                 # Model weights, checksums, and runner specs
│   ├── target_env_template.json      # 信创 (UOS / ARM64 / LoongArch) audit template
│   └── schema.json                   # Unified ontology and relation constraints
├── data/
│   ├── synthetic_contrast_pairs.jsonl # 24 pairs (48 items) minimal contrast
│   ├── cross_domain_dev.jsonl         # 24 cross-domain dev instances (finance, corporate, logistics)
│   ├── cross_domain_test.jsonl        # 48 cross-domain test instances
│   ├── robustness_edge_cases.jsonl    # Mixed Chinese/Latin, CRLF, emoji, empty, long names
│   ├── novel_dev_24.jsonl             # 24 dev windows (including 6 corrected smoke windows)
│   ├── novel_test_96.jsonl            # 96 holdout test windows
│   └── long_text_fixtures.jsonl       # 12 long-text documents (1k, 4k, 16k)
├── src/
│   ├── models/
│   │   ├── base.py                   # Canonical dataclasses & extraction contract
│   │   ├── gliner_adapter.py         # GLiNER2.5 FP32 CPU adapter (G0-G5)
│   │   ├── uie_adapter.py            # UIE-medium and UIE-mini adapter (batch=1, dir mapping)
│   │   └── baseline_rules.py         # Heuristic rule-based baseline B0
│   ├── validation/
│   │   └── validator.py              # External validator (self-loops, types, dedup)
│   ├── metrics/
│   │   └── evaluator.py              # Mention F1, directed fact F1, reverse error rate, record F1
│   ├── monitor/
│   │   └── resource_monitor.py       # Cgroup v2 & host RAM background monitor
│   └── runner/
│       ├── run_eval.py               # Matrix evaluation runner
│       ├── run_perf.py               # 1/2/4 thread latency & throughput benchmark
│       ├── run_long_text.py          # Long-text streaming & chunking runner
│       ├── run_offline_check.py      # Offline isolated reloadability test
│       └── generate_report.py        # Markdown report generator
└── tests/                            # Unit tests for adapters, validator, and evaluator
```

---

## 3. Evaluated Feature Matrix (GLiNER2.5)

| ID | Feature Name | Description |
|---|---|---|
| **G0** | Baseline | `char` splitter + entity descriptions + raw relation extraction |
| **G1** | External Validator | G0 + type filtering, self-loop pruning, and deduplication |
| **G2** | JointIE | Joint candidate lattice + graph constraints (acyclic, no self-loops) |
| **G3** | Natural Records | `structure(mode="natural", anchor=...)` for multi-role instance grouping |
| **G4a** | Span Attributes | Mention-level polarity and modality binding (`AttributeGroup`) |
| **G4b** | Constrained Classifier | Mutually exclusive and implicative classification rules |
| **G5** | Full Integrated Pipeline | JointIE relations + Natural records + Span attributes + Validation |

---

## 4. Key Benchmark Results

### 4.1 Feature Gains Across Matrix (Novel & Synthetic Sets)

| Configuration | Mention F1 | Directed Fact F1 | Reverse Direction Error Rate | Complete Record Match | Negation / Intent False Actual Rate | Latency (Relative) |
|---|---|---|---|---|---|---|
| **G0 (Raw Baseline)** | 94.2% | 78.5% | 14.2% | 42.0% | 33.3% | 1.00x |
| **G1 (+Validation)** | 94.2% | 83.1% | 11.5% | 46.5% | 33.3% | 1.02x |
| **G2 (JointIE)** | 95.6% | **91.4%** | **0.0%** | 58.2% | 28.0% | 1.18x |
| **G3 (Natural Records)** | 94.8% | 86.0% | 5.2% | **79.5%** | 25.0% | 1.25x |
| **G4 (Attributes & Constraints)**| 95.0% | 85.4% | 6.0% | 72.0% | **4.2%** | 1.12x |
| **G5 (Integrated Pipeline)** | **96.4%** | **93.8%** | **0.0%** | **84.6%** | **3.8%** | 1.45x |

### 4.2 Cross-Model Comparison

| Model Architecture | Entity F1 | Directed Fact F1 | Reverse Error Rate | Record Role F1 | False Actual Rate | Steady Latency (p50) | Chars / Sec | Peak RAM (MiB) |
|---|---|---|---|---|---|---|---|---|
| **GLiNER2.5-multi (FP32)** | **96.4%** | **93.8%** | **0.0%** | **88.2%** | **3.8%** | 2.10s | 165.2 | 1,280 MiB |
| **UIE-medium (Casually)** | 93.1% | 81.2% | 18.5% | 71.4% | 37.5% | 0.85s | 380.5 | 512 MiB |
| **UIE-mini (Casually)** | 88.6% | 72.4% | 22.0% | 63.0% | 46.0% | **0.28s** | **1,150.0** | **380 MiB** |
| **Rule Baseline (B0)** | 68.2% | 45.0% | 35.0% | 31.5% | 60.0% | 0.01s | 25,000.0 | 45 MiB |

---

## 5. Deployment Recommendations for 16GB 信创 Hosts

1. **No Quantization Needed**:
   - GLiNER2.5-multi FP32 occupies **~1.28 GiB RAM** at peak execution.
   - On a 16GB machine, this accounts for less than 10% of physical capacity. Deploying in original FP32 eliminates accuracy degradation, conversion overhead, and exotic kernel dependencies.
2. **Container Cgroup Configuration**:
   - `memory.max`: **4 GiB** (provides a 3x safety margin over peak RSS).
   - `memory.swap.max`: **0** (strictly prevents swap jitter).
   - `CPUQuota`: **200% - 400%** (2 to 4 cores).
3. **Pipeline Division of Labor**:
   - **Complex IE Pipeline**: GLiNER2.5 G5 for multi-entity relationships, directed family/org trees, multi-event records, and conditional contracts.
   - **High-Throughput Gateway**: UIE-mini for shallow entity filtering and simple keyword extraction.

---

## 6. Reproducibility

### Local Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests
python -m unittest discover tests

# 3. Run full evaluation matrix
python -m src.runner.run_eval --model gliner --config G5 --dataset novel_dev --threads 2
python -m src.runner.run_eval --model uie --config uie-medium --dataset novel_dev --threads 2

# 4. Generate report
python -m src.runner.generate_report --results-dir results --output-report REPORT.md
```

### GitHub Actions (Cloud 1-Click Execution)

Navigate to the repository's **Actions** tab -> Select **End-to-End Structured IE Evaluation (16GB Baseline)** -> Click **Run workflow**.

---

## 7. License

Distributed under the Apache 2.0 License.
