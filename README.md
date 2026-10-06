*This project has been created as part of the 42 curriculum by laoubaid.*

# Call Me Maybe - Function Calling with Constrained Decoding

## Description

**Call Me Maybe** is an introduction to function calling in Large Language Models (LLMs). Small parameter models (such as `Qwen/Qwen3-0.6B`) are notoriously unreliable when tasked with generating structured JSON outputs purely through natural language prompting. They frequently deviate from schemas, hallucinate keys, or produce syntactically broken JSON.

This project bridges that gap by implementing **Constrained Decoding** (logit masking) at the token generation level. Rather than hoping the model outputs valid structure, we actively intervene during the autoregressive generation loop: invalid tokens are masked with negative infinity (`-inf`), ensuring that the model is mathematically restricted to selecting only tokens that satisfy the required function signatures and JSON schema.

---

## Instructions

### Prerequisites
- Python >= 3.10 (Project configured for Python 3.12)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

Clone the repository and install dependencies using `uv`:

```bash
uv sync
```

Alternatively, using the provided `Makefile`:

```bash
make install
```

### Execution

Run the project through the standard package execution:

```bash
uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
```

Or using the installed script entry point:

```bash
uv run callme [--functions_definition <path>] [--input <path>] [--output <path>]
```

By default, the program reads input files from `data/input/` and writes results to `data/output/`:
- **Default input prompts**: `data/input/function_calling_tests.json`
- **Default functions definition**: `data/input/functions_definition.json`
- **Default output destination**: `data/output/function_calling_results.json`

---

## Example Usage

Run with default files:
```bash
uv run python -m src
```

Run with custom test files:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Example input prompt:
```json
{
  "prompt": "What is the sum of 40 and 2?"
}
```

Generated output:
```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40,
    "b": 2
  }
}
```

---

## Algorithm Explanation

### Demystifying the Black Box: How LLMs Turn Words into Math

![Demystifying the Black Box: How LLMs Turn Words into Math](data/figures/LLMs%20as%20spreadsheets.png)

Language models operate by computing a probability distribution over their entire vocabulary for each successive token. Unconstrained decoding samples or picks the `argmax` from this raw distribution.

In **Constrained Decoding**, we apply a structural mask over the vocabulary at every step:
1. Identify all tokens that maintain valid JSON syntax and match the expected schema.
2. Set the logits of all other tokens to $-\infty$.
3. Compute the softmax over the filtered logits, forcing the model to select only legal candidates.

### Generation Pipeline

The generation pipeline follows these exact steps:

![Constrained Decoding Pipeline via Logit Masking](data/figures/Constrained%20Decoding%20Pipeline.png)

1. **Prompt**: The natural language user query combined with function definitions.
2. **Encode**: Convert the prompt string into subword token representations.
3. **Input IDs**: Numerical vector representation consumed by the neural network.
4. **Processing**: Forward pass through the model's transformer layers.
5. **Raw Logits**: Unnormalized scores over the vocabulary for the next token position.
6. **Filtering (Masking)**: Compute valid next token IDs given the current grammar/prefix state and set invalid logits to `-inf`.
7. **Select Next Token ID**: Pick the most probable valid token (`argmax` over masked logits).
8. **Append to Input IDs**: Append the selected token to the sequence and feed back into Step 3 until the complete JSON is generated.
9. **Decode**: Convert the generated token IDs back into text and parse into the validated schema.

### Function Name & Schema Prefix Paths

To constrain the function name, we pre-encode each candidate function name into token paths:
- `fn_add_numbers` $\rightarrow$ `[8822, 2891, 32964]`
- `fn_greet` $\rightarrow$ `[8822, 1889, 3744]`

At step $t$, only tokens that form valid prefixes of candidate paths are allowed. Once the function name is resolved, its parameters and type constraints are looked up to guide subsequent parameter decoding.
