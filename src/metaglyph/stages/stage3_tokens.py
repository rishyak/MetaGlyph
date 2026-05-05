"""Stage 3: Token Accounting & Matching.

This module validates and matches token counts across prompt conditions.
The key insight is that NL and MG instructions must have matched token counts
to isolate semantic effects from length effects.

Token matching strategy:
- NL ↔ MG: Match instruction tokens (±1 tolerance)
- CTRL ↔ MG: Match instruction tokens (±1 tolerance)

If matching fails:
- Regenerate NL paraphrase with padding/trimming
- Adjust MG formatting (whitespace, parentheses)

Token accounting strategy:
- Report instruction-only and full-prompt token counts separately.
- Compare control instruction lengths against MG (± tolerance).
- Do not rewrite prompt artifacts during validation.

Artifacts produced:
- tokens/<model>/<prompt_id>.json
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable
import re

from ..conditions import CONDITIONS, CONTROL_CONDITIONS, PromptCondition
from ..utils.io_utils import load_json, save_json, load_text, ensure_dir, list_files


@dataclass
class TokenCounts:
    """Token counts for a prompt."""

    prompt_id: str
    model: str
    tokenizer: str
    condition: str
    instruction_tokens: int
    input_tokens: int
    output_format_tokens: int
    total_tokens: int
    validated: bool
    adjustments: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class Tokenizer:
    """Abstract tokenizer interface supporting multiple backends."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        raise NotImplementedError

    def tokenize(self, text: str) -> list[str]:
        """Return tokens as strings."""
        raise NotImplementedError


class TiktokenTokenizer(Tokenizer):
    """Tokenizer using tiktoken (OpenAI tokenizers)."""

    MODEL_ENCODINGS = {
        "o200k_base": "o200k_base",
        "cl100k_base": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-5.2-chat": "o200k_base",
        "openai/gpt-5.2-chat": "o200k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-davinci-003": "p50k_base",
    }

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._load_error = None
        try:
            import tiktoken

            encoding_key = model_name.removeprefix("tiktoken:")
            encoding_name = self.MODEL_ENCODINGS.get(encoding_key, encoding_key)
            self._tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            self._load_error = e
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            raise RuntimeError(
                f"Could not load tiktoken tokenizer '{self.model_name}'. "
                "Install the tokenizers dependency group with `uv sync --group tokenizers`."
            ) from self._load_error
        return len(self._tokenizer.encode(text))

    def tokenize(self, text: str) -> list[str]:
        if self._tokenizer is None:
            raise RuntimeError(
                f"Could not load tiktoken tokenizer '{self.model_name}'."
            ) from self._load_error
        tokens = self._tokenizer.encode(text)
        return [self._tokenizer.decode([t]) for t in tokens]


class TransformersTokenizer(Tokenizer):
    """Tokenizer using HuggingFace transformers."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._load_error = None
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            self._load_error = e
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            raise RuntimeError(
                f"Could not load HuggingFace tokenizer '{self.model_name}'. "
                "Install the tokenizers dependency group and verify any required model access."
            ) from self._load_error
        return len(self._tokenizer.encode(text))

    def tokenize(self, text: str) -> list[str]:
        if self._tokenizer is None:
            raise RuntimeError(
                f"Could not load HuggingFace tokenizer '{self.model_name}'."
            ) from self._load_error
        return self._tokenizer.tokenize(text)


class SimpleTokenizer(Tokenizer):
    """Simple whitespace-based tokenizer for testing."""

    def __init__(self, model_name: str = "simple"):
        super().__init__(model_name)

    def count_tokens(self, text: str) -> int:
        # Approximate: split on whitespace and punctuation
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return len(tokens)

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text)


class TokenMatcherError(Exception):
    """Raised when token matching cannot be achieved."""

    pass


class InstructionAdjuster:
    """Adjusts instructions to match target token counts."""

    # Padding phrases that can be added/removed without changing semantics
    NL_PADDING_PHRASES = [
        "Please note that ",
        "It is important to ",
        "Make sure to carefully ",
        "You should ",
        "Be sure to ",
        "Take care to ",
        "Remember to ",
        "Ensure that you ",
    ]

    # Filler words that can be added/removed
    NL_FILLERS = [
        "carefully",
        "thoroughly",
        "properly",
        "correctly",
        "accurately",
        "precisely",
    ]

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def adjust_nl_to_target(
        self,
        instruction: str,
        target_tokens: int,
        tolerance: int = 1,
        max_iterations: int = 50,
    ) -> tuple[str, list[str]]:
        """
        Adjust NL instruction to match target token count.

        Returns adjusted instruction and list of adjustments made.
        """
        adjustments = []
        current = instruction
        current_tokens = self.tokenizer.count_tokens(current)

        for _ in range(max_iterations):
            diff = current_tokens - target_tokens

            if abs(diff) <= tolerance:
                return current, adjustments

            if diff > 0:
                # Need to remove tokens
                current, adj = self._remove_tokens(current, diff)
            else:
                # Need to add tokens
                current, adj = self._add_tokens(current, -diff)

            if adj:
                adjustments.append(adj)
            else:
                break  # No more adjustments possible

            current_tokens = self.tokenizer.count_tokens(current)

        # Final check
        if abs(self.tokenizer.count_tokens(current) - target_tokens) <= tolerance:
            return current, adjustments

        raise TokenMatcherError(
            f"Could not match tokens: target={target_tokens}, "
            f"achieved={self.tokenizer.count_tokens(current)}"
        )

    def _remove_tokens(self, text: str, count: int) -> tuple[str, str]:
        """Remove approximately 'count' tokens from text."""
        # Try removing filler words
        for filler in self.NL_FILLERS:
            pattern = rf"\b{filler}\b\s*"
            if re.search(pattern, text, re.IGNORECASE):
                new_text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
                return new_text.strip(), f"removed '{filler}'"

        # Try removing padding phrases
        for phrase in self.NL_PADDING_PHRASES:
            if phrase.lower() in text.lower():
                new_text = text.replace(phrase, "", 1)
                return new_text.strip(), f"removed '{phrase.strip()}'"

        # Remove extra whitespace
        new_text = re.sub(r"\s+", " ", text).strip()
        if new_text != text:
            return new_text, "normalized whitespace"

        return text, ""

    def _add_tokens(self, text: str, count: int) -> tuple[str, str]:
        """Add approximately 'count' tokens to text."""
        import random

        # Add filler words
        if count <= 2:
            filler = random.choice(self.NL_FILLERS)
            # Insert after "Please" or at start
            if "Please " in text:
                new_text = text.replace("Please ", f"Please {filler} ", 1)
            else:
                new_text = f"Please {filler} " + text[0].lower() + text[1:]
            return new_text, f"added '{filler}'"

        # Add padding phrase
        phrase = random.choice(self.NL_PADDING_PHRASES)
        # Find a good insertion point (after first sentence or clause)
        match = re.search(r"[.!?]\s+", text)
        if match:
            pos = match.end()
            new_text = text[:pos] + phrase + text[pos].lower() + text[pos + 1 :]
        else:
            new_text = phrase + text[0].lower() + text[1:]

        return new_text, f"added '{phrase.strip()}'"

    def adjust_mg_formatting(
        self,
        instruction: str,
        target_tokens: int,
        tolerance: int = 1,
    ) -> tuple[str, list[str]]:
        """
        Adjust MG instruction formatting to match target.

        MG adjustments are limited to:
        - Whitespace around operators
        - Parentheses grouping
        - Symbol spacing
        """
        adjustments = []
        current = instruction
        current_tokens = self.tokenizer.count_tokens(current)

        diff = current_tokens - target_tokens

        if abs(diff) <= tolerance:
            return current, adjustments

        if diff > 0:
            # Remove spaces around operators
            for op in ["→", "↦", "∈", "∉", "∩", "∪", "⇒", "∘", "|"]:
                if f" {op} " in current:
                    current = current.replace(f" {op} ", f"{op}", 1)
                    adjustments.append(f"removed spaces around '{op}'")
                    if (
                        self.tokenizer.count_tokens(current) - target_tokens
                        <= tolerance
                    ):
                        break
        else:
            # Add spaces around operators
            for op in ["→", "↦", "∈", "∉", "∩", "∪", "⇒", "∘", "|"]:
                if op in current and f" {op} " not in current:
                    current = current.replace(op, f" {op} ", 1)
                    adjustments.append(f"added spaces around '{op}'")
                    if (
                        target_tokens - self.tokenizer.count_tokens(current)
                        <= tolerance
                    ):
                        break

        final_tokens = self.tokenizer.count_tokens(current)
        if abs(final_tokens - target_tokens) <= tolerance:
            return current, adjustments

        # If still not matching, accept with note
        adjustments.append(f"residual diff: {final_tokens - target_tokens}")
        return current, adjustments


class TokenMatcher:
    """Main class for token accounting and matching."""

    def __init__(
        self,
        prompts_dir: Path | str,
        tasks_dir: Path | str,
        output_dir: Path | str,
        model_name: str = "simple",
        output_label: str | None = None,
        tolerance: int = 1,
    ):
        self.prompts_dir = Path(prompts_dir)
        self.tasks_dir = Path(tasks_dir)
        self.output_dir = Path(output_dir)
        self.tolerance = tolerance

        # Initialize tokenizer based on explicit backend/model selection.
        if model_name == "simple":
            self.tokenizer = SimpleTokenizer(model_name)
        elif (
            model_name.startswith("tiktoken:")
            or model_name in TiktokenTokenizer.MODEL_ENCODINGS
        ):
            self.tokenizer = TiktokenTokenizer(model_name)
        elif "/" in model_name:  # HuggingFace model path
            self.tokenizer = TransformersTokenizer(model_name)
        else:
            raise ValueError(
                f"Unknown tokenizer_model '{model_name}'. Use 'simple', "
                "'o200k_base', 'cl100k_base', 'tiktoken:<encoding>', or a HuggingFace tokenizer ID."
            )

        self.adjuster = InstructionAdjuster(self.tokenizer)
        self.model_name = model_name
        self.output_label = output_label or model_name

    def validate_all(self) -> dict[str, list[TokenCounts]]:
        """Validate and match tokens for all prompts."""
        all_counts = {}

        for family_dir in self.prompts_dir.iterdir():
            if not family_dir.is_dir():
                continue

            family_name = family_dir.name
            counts = self._validate_family(family_name)
            all_counts[family_name] = counts

        return all_counts

    def validate_family(self, family_name: str) -> list[TokenCounts]:
        """Validate and match tokens for a specific family."""
        return self._validate_family(family_name)

    def _validate_family(self, family_name: str) -> list[TokenCounts]:
        """Internal method for family validation.

        Reads the per-instance prompt metadata generated by Stage 2 and counts
        both instruction-only and full-prompt tokens. This keeps instruction
        compression distinct from end-to-end prompt cost.
        """
        prompts_family_dir = self.prompts_dir / family_name
        output_dir = self.output_dir / self.output_label / family_name
        ensure_dir(output_dir)

        all_counts = []

        prompt_files = sorted(list_files(prompts_family_dir, "*.json"))
        if not prompt_files:
            raise TokenMatcherError(
                f"No Stage 2 prompt metadata found in {prompts_family_dir}. "
                "Run Stage 2 before token validation."
            )

        prompts_by_instance: dict[str, dict[str, dict[str, Any]]] = {}
        for prompt_file in prompt_files:
            prompt = load_json(prompt_file)
            instance_id = prompt.get("instance_id")
            condition = prompt.get("condition")
            if not instance_id or condition not in CONDITIONS:
                continue
            prompts_by_instance.setdefault(instance_id, {})[condition] = prompt

        for instance_id, prompts in sorted(prompts_by_instance.items()):
            counts = self._validate_instance(instance_id, prompts, output_dir)
            all_counts.extend(counts)

        if not all_counts:
            raise TokenMatcherError(f"No valid prompts found in {prompts_family_dir}")

        return all_counts

    def _validate_instance(
        self,
        instance_id: str,
        prompts: dict[str, dict[str, Any]],
        output_dir: Path,
    ) -> list[TokenCounts]:
        """Validate and match tokens for a single instance."""
        results = []

        # Get MG instruction as reference for token matching
        mg_instruction = prompts.get(PromptCondition.MG.value, {}).get(
            "instruction", ""
        )
        mg_tokens = self.tokenizer.count_tokens(mg_instruction)

        for condition, prompt in sorted(prompts.items()):
            instruction = prompt.get("instruction", "")
            input_text = prompt.get("input_text", "")
            output_format = prompt.get("output_format", "")
            full_prompt = prompt.get("full_prompt", "")

            instr_tokens = self.tokenizer.count_tokens(instruction)
            input_tokens = self.tokenizer.count_tokens(input_text)
            output_format_tokens = self.tokenizer.count_tokens(output_format)
            total_tokens = self.tokenizer.count_tokens(full_prompt)

            # Check if matching is needed
            adjustments = []
            validated = True

            if condition in CONTROL_CONDITIONS:
                # Controls are expected to be token-matched against MG.
                # Stage 3 reports the actual prompts on disk; it does not claim
                # hypothetical rewrites unless a prompt artifact is changed.
                diff = abs(instr_tokens - mg_tokens)
                if diff > self.tolerance:
                    adjustments = [
                        f"{condition} instruction tokens differ from MG by {diff}; "
                        "prompt artifact was not rewritten"
                    ]
                    validated = False

            prompt_id = f"{instance_id}_{condition}"
            token_counts = TokenCounts(
                prompt_id=prompt_id,
                model=self.output_label,
                tokenizer=self.model_name,
                condition=condition,
                instruction_tokens=instr_tokens,
                input_tokens=input_tokens,
                output_format_tokens=output_format_tokens,
                total_tokens=total_tokens,
                validated=validated,
                adjustments=adjustments,
            )

            results.append(token_counts)

            # Save token data
            save_json(token_counts.to_dict(), output_dir / f"{prompt_id}.json")

        return results

    def get_compression_stats(self, family_name: str) -> dict[str, Any]:
        """Calculate compression statistics for a family."""
        prompts_family_dir = self.prompts_dir / family_name

        by_condition = {
            condition: {"instruction": [], "total": []} for condition in CONDITIONS
        }
        for prompt_file in sorted(list_files(prompts_family_dir, "*.json")):
            prompt = load_json(prompt_file)
            condition = prompt.get("condition")
            if condition not in by_condition:
                continue
            by_condition[condition]["instruction"].append(
                self.tokenizer.count_tokens(prompt.get("instruction", ""))
            )
            by_condition[condition]["total"].append(
                self.tokenizer.count_tokens(prompt.get("full_prompt", ""))
            )

        def avg(values: list[int]) -> float:
            return sum(values) / len(values) if values else 0.0

        nl_tokens = avg(by_condition["NL"]["instruction"])
        mg_tokens = avg(by_condition["MG"]["instruction"])
        ctrl_tokens = avg(by_condition["CTRL"]["instruction"])
        nl_total_tokens = avg(by_condition["NL"]["total"])
        mg_total_tokens = avg(by_condition["MG"]["total"])
        ctrl_total_tokens = avg(by_condition["CTRL"]["total"])

        instruction_compression_ratio = mg_tokens / nl_tokens if nl_tokens > 0 else 0
        full_prompt_compression_ratio = (
            mg_total_tokens / nl_total_tokens if nl_total_tokens > 0 else 0
        )

        stats = {
            "family": family_name,
            "avg_nl_instruction_tokens": nl_tokens,
            "avg_mg_instruction_tokens": mg_tokens,
            "avg_ctrl_instruction_tokens": ctrl_tokens,
            "avg_nl_total_tokens": nl_total_tokens,
            "avg_mg_total_tokens": mg_total_tokens,
            "avg_ctrl_total_tokens": ctrl_total_tokens,
            "instruction_compression_ratio": instruction_compression_ratio,
            "full_prompt_compression_ratio": full_prompt_compression_ratio,
        }

        return stats
