# MetaGlyph — Symbolic Metalanguages for LLM Prompting

This repository contains the research artifacts, datasets, and evaluation framework for studying **symbolic metalanguages for large language model (LLM) prompting**. The goal of this project is to evaluate whether mathematical and logical operators can be used to **semantically compress instruction language**, independent of context compression or learned prompt optimization.

The work is designed to be **fully automatic, reproducible, and long-context aware**, using **free-tier models via OpenRouter API**.

---

## Quick start

### 1. Setup

```bash
uv sync

# Optional: install the complete research/reproduction environment
uv sync --group all
```

### 2. Configure API key

Create a `.env` file with your OpenRouter API key:

```
OPENROUTER_API_KEY=your_api_key_here
```

Get a free API key at [openrouter.ai](https://openrouter.ai).

### 3. Generate task instances and prompts

```bash
# Generate tasks, prompts, and token accounting without model calls
uv run metaglyph --stage 1-3
```

### 4. Run experiments

```bash
# Run model execution, evaluation, and aggregation
uv run metaglyph --stage 4-6

# Run only model execution (Stage 4)
uv run metaglyph --stage 4

# Re-run evaluation and reporting only (Stage 5-6)
uv run metaglyph --stage 5,6
```

---

## Models

Experiments use models via OpenRouter API:

| Model | OpenRouter ID | Size | Purpose |
|-------|---------------|------|---------|
| Llama 3.2 3B Instruct | `meta-llama/llama-3.2-3b-instruct` | 3B | Small dense baseline |
| Gemma 3 12B | `google/gemma-3-12b-it` | 12B | Mid-size instruction follower |
| Qwen 2.5 7B Instruct | `qwen/qwen-2.5-7b-instruct` | 7B | Multi-lingual instruction model |
| Qwen3 4B | `qwen/qwen3-4b` | 4B | Compact instruction model |
| OLMo 3 7B Instruct | `allenai/olmo-3-7b-instruct` | 7B | Fully open-weight baseline (AI2) |
| OLMo 3 32B Instruct | `allenai/olmo-3-32b-instruct` | 32B | Large open-weight model |
| Kimi K2 | `moonshotai/kimi-k2` | 1T (32B active) | MoE wildcard with agentic tuning |
| Gemini 2.5 Flash | `google/gemini-2.5-flash-preview` | - | Fast multimodal model |
| Claude Haiku 4.5 | `anthropic/claude-haiku-4.5` | - | Fast instruction follower |
| GPT-5.2 Chat | `openai/gpt-5.2-chat` | - | OpenAI frontier model |

**Note:** Requires OpenRouter API key.

### Execution parameters

All models run with identical settings:
- `temperature: 0` (deterministic)
- `top_p: 1.0`
- `frequency_penalty: 0`
- `presence_penalty: 0`
- `max_tokens: 2048`

---

## Pipeline architecture

The pipeline has **six stages**, executed in order:

```
Stage 1: Dataset & Task Specification
    ↓
Stage 2: Prompt Construction (NL / NL_SHORT / ASCII_DSL / MG / CTRL / CTRL_RANDOM)
    ↓
Stage 3: Token Accounting & Matching
    ↓
Stage 4: Model Execution ← Only stage using LLMs
    ↓
Stage 5: Automatic Evaluation
    ↓
Stage 6: Aggregation & Reporting
```

### Stage outputs

| Stage | Artifacts |
|-------|-----------|
| 1 | `tasks/<family>/*.{input,gold,constraints,meta}` |
| 2 | `prompts/<family>/*.txt` |
| 3 | `tokens/<model>/*.json` |
| 4 | `outputs/<model>/*.txt`, `runs/<model>/*.meta` |
| 5 | `results/<model>/*.json` |
| 6 | `summary/tables/*.csv`, `summary/figures/*.pdf` |

---

## Project motivation

Natural-language prompts act as *instruction languages*, specifying how inputs should be selected, transformed, or constrained. While effective, natural language is verbose and ambiguous as a control channel. This project investigates whether **symbolic operators already internalized during model pretraining** (e.g., ∈, ¬, ∩, ⇒) can function as compact, reliable instruction-semantic primitives.

Unlike prompt compression systems that prune context, or constructed prompt languages that rely on system-level decoding schemes, this work focuses on **semantic compression of the instruction language itself**, under strict token control.

---

## Task families

Four task families, each testing different operator semantics:

| Family | Operators | Description |
|--------|-----------|-------------|
| Selection & Classification | ∈, ∉, ¬, ∩, ∪ | Select items based on set membership |
| Structured Extraction | ∈, →, ↦, \| | Extract fields from documents |
| Constraint Composition | ∩, ∪, ¬, ⊆, ∀, ∃ | Apply composed constraints |
| Conditional Transformation | ⇒, ∘, \|, → | Transform based on rules |

---

## Verification metrics

The pipeline reports instruction-token compression separately from full-prompt compression. This matters because the input document and output-format wrapper are identical across conditions, so a large instruction-only reduction can be a much smaller end-to-end prompt reduction.

Stage 3 writes per-prompt token records with:
- `instruction_tokens`
- `input_tokens`
- `output_format_tokens`
- `total_tokens`

The default configuration uses `o200k_base` via `tiktoken` for reproducible OpenAI-style counts. For provider billing or latency claims, compare Stage 3 counts with provider-reported `usage.prompt_tokens` from the actual API response.

---

## Experimental design

Each task instance is evaluated under six instruction conditions:

1. **NL** — verbose natural-language instruction
2. **NL_SHORT** — compact natural-language baseline
3. **ASCII_DSL** — SQL/code-like ASCII pseudocode baseline
4. **MG** — compact MetaGlyph symbolic instruction
5. **CTRL** — swapped-operator control with broken semantics
6. **CTRL_RANDOM** — same-shape random-symbol control

The key comparison is not only MG versus verbose prose. Review MG against `NL_SHORT` and `ASCII_DSL` to test whether symbolic Unicode is actually better than terse English or familiar pseudocode.

Top-line correctness uses strict exact match against gold outputs. Operator-fidelity checks are reported as diagnostics only and do not turn partial or under-specified answers into passing results.

---

## Symbolic operator inventory

MetaGlyph uses high-frequency mathematical and logical operators:

| Category | Operators |
|----------|-----------|
| Transformation | `→`, `⇒`, `∘`, `↦` |
| Set/constraints | `∈`, `∉`, `⊆`, `∩`, `∪` |
| Logical | `¬`, `∀`, `∃` |
| Scope | `\|` |

---

## CLI reference

```bash
# Full pipeline
uv run metaglyph

# Specific stages
uv run metaglyph --stage 1        # Dataset generation only
uv run metaglyph --stage 1-3      # Stages 1 through 3
uv run metaglyph --stage 4,5,6    # Execution + evaluation

# Configuration
uv run metaglyph --instances 50   # 50 instances per family
uv run metaglyph --models llama-3.2-3b,qwen-2.5-7b
uv run metaglyph --backend openrouter
uv run metaglyph --config custom.json

# With custom config file
uv run metaglyph --config my_config.json
```

---

## Reproducibility

- All experiments use models via OpenRouter API
- Model IDs, decoding parameters, and seeds are fixed
- Results can be regenerated end-to-end
- No manual inspection required for scoring
- Stage 3 and Stage 4 both consume the exact per-instance prompt files emitted by Stage 2
- Stage 5 requires exact matches for `overall_pass`; partial metrics remain diagnostic
- Token reports distinguish instruction-only savings from full-prompt savings

---

## Scope and limitations

This repository focuses on **instruction semantics**, not system performance. It does **not** claim or measure:

- Latency improvements
- Memory usage
- Attention complexity
- Throughput or cost savings

The experiments are single-turn and do not evaluate multi-turn dialogue.

---

## Citation

```bibtex
@article{metaglyph2025,
  title={Semantic Compression of LLM Instructions via Symbolic Metalanguages},
  author={Ernst van Gassen},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

This repository is released for research and academic use. See the `LICENSE` file for details.
