"""Stage 4: Model Execution.

This module runs validated prompts through language models.
It is the ONLY stage where models are actually invoked.

Execution settings:
- Deterministic decoding (temperature=0)
- Fixed context length
- Fixed output token limit
- Same settings across NL/MG/CTRL conditions

Supported backends:
- Ollama (local, primary)
- OpenAI API (optional validation)
- Anthropic API (optional validation)

Artifacts produced:
- outputs/<model>/<prompt_id>.txt  - Raw model output
- runs/<model>/<prompt_id>.meta    - Runtime metadata
"""

import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
from abc import ABC, abstractmethod
import hashlib

from ..conditions import CONDITIONS, get_prompt_condition
from ..utils.io_utils import load_json, load_text, save_json, save_text, ensure_dir, list_files


@dataclass
class ExecutionConfig:
    """Configuration for model execution."""
    temperature: float = 0.0  # Deterministic
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 781
    max_context_tokens: int = 8192
    timeout_seconds: int = 120
    seed: Optional[int] = 42  # For reproducibility where supported
    request_delay: float = 0.5  # Delay between requests (seconds) to respect rate limits
    max_workers: int = 8


@dataclass
class ExecutionResult:
    """Result of a single model execution."""
    prompt_id: str
    model: str
    raw_output: str
    success: bool
    error_message: Optional[str]
    execution_time_ms: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    config: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunMetadata:
    """Metadata for a model run."""
    prompt_id: str
    model: str
    model_version: Optional[str]
    timestamp: str
    execution_time_ms: int
    config: dict
    prompt_hash: str
    success: bool
    error: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    def __init__(self, model_name: str, config: ExecutionConfig):
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def generate(self, prompt: str) -> ExecutionResult:
        """Generate a response for the given prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass

    def get_model_version(self) -> Optional[str]:
        """Get the specific model version if available."""
        return None


class OllamaBackend(ModelBackend):
    """Backend for local Ollama models."""

    def __init__(self, model_name: str, config: ExecutionConfig, base_url: str = "http://localhost:11434"):
        super().__init__(model_name, config)
        self.base_url = base_url

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m.get("name", "").startswith(self.model_name) for m in models)
            return False
        except Exception:
            return False

    def generate(self, prompt: str) -> ExecutionResult:
        """Generate using Ollama API."""
        import requests

        start_time = time.time()
        prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "repeat_penalty": 1.0 + self.config.frequency_penalty,  # Ollama equivalent
                        "num_predict": self.config.max_tokens,
                        "num_ctx": self.config.max_context_tokens,
                        "seed": self.config.seed,
                    },
                },
                timeout=self.config.timeout_seconds,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                result = response.json()
                return ExecutionResult(
                    prompt_id=prompt_id,
                    model=self.model_name,
                    raw_output=result.get("response", ""),
                    success=True,
                    error_message=None,
                    execution_time_ms=elapsed_ms,
                    input_tokens=result.get("prompt_eval_count"),
                    output_tokens=result.get("eval_count"),
                    config=asdict(self.config),
                )
            else:
                return ExecutionResult(
                    prompt_id=prompt_id,
                    model=self.model_name,
                    raw_output="",
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.text}",
                    execution_time_ms=elapsed_ms,
                    input_tokens=None,
                    output_tokens=None,
                    config=asdict(self.config),
                )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_name,
                raw_output="",
                success=False,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )

    def get_model_version(self) -> Optional[str]:
        """Get Ollama model version."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for m in models:
                    if m.get("name", "").startswith(self.model_name):
                        return m.get("digest", "")[:12]
        except Exception:
            pass
        return None


class OpenAIBackend(ModelBackend):
    """Backend for OpenAI API models."""

    def __init__(self, model_name: str, config: ExecutionConfig, api_key: Optional[str] = None):
        super().__init__(model_name, config)
        self.api_key = api_key

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.api_key:
            import os
            self.api_key = os.environ.get("OPENAI_API_KEY")
        return bool(self.api_key)

    def generate(self, prompt: str) -> ExecutionResult:
        """Generate using OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            return ExecutionResult(
                prompt_id="",
                model=self.model_name,
                raw_output="",
                success=False,
                error_message="openai package not installed",
                execution_time_ms=0,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )

        start_time = time.time()
        prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                max_tokens=self.config.max_tokens,
                seed=self.config.seed,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_name,
                raw_output=response.choices[0].message.content or "",
                success=True,
                error_message=None,
                execution_time_ms=elapsed_ms,
                input_tokens=response.usage.prompt_tokens if response.usage else None,
                output_tokens=response.usage.completion_tokens if response.usage else None,
                config=asdict(self.config),
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_name,
                raw_output="",
                success=False,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )


class AnthropicBackend(ModelBackend):
    """Backend for Anthropic API models."""

    def __init__(self, model_name: str, config: ExecutionConfig, api_key: Optional[str] = None):
        super().__init__(model_name, config)
        self.api_key = api_key

    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        if not self.api_key:
            import os
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        return bool(self.api_key)

    def generate(self, prompt: str) -> ExecutionResult:
        """Generate using Anthropic API."""
        try:
            import anthropic
        except ImportError:
            return ExecutionResult(
                prompt_id="",
                model=self.model_name,
                raw_output="",
                success=False,
                error_message="anthropic package not installed",
                execution_time_ms=0,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )

        start_time = time.time()
        prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                messages=[{"role": "user", "content": prompt}],
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            output_text = response.content[0].text if response.content else ""

            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_name,
                raw_output=output_text,
                success=True,
                error_message=None,
                execution_time_ms=elapsed_ms,
                input_tokens=response.usage.input_tokens if response.usage else None,
                output_tokens=response.usage.output_tokens if response.usage else None,
                config=asdict(self.config),
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_name,
                raw_output="",
                success=False,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )


class OpenRouterBackend(ModelBackend):
    """Backend for OpenRouter API (supports many models via unified API)."""

    # Model ID mapping for convenience (paid versions - no :free suffix)
    MODEL_IDS = {
        # Smaller models
        "llama-3.2-3b": "meta-llama/llama-3.2-3b-instruct",
        "qwen3-4b": "qwen/qwen3-4b",
        "gemma-3-12b": "google/gemma-3-12b-it",
        # Open / larger models
        "olmo-3.1-32b-instruct": "allenai/olmo-3.1-32b-instruct",
        "kimi-k2": "moonshotai/kimi-k2",
        "llama-4-maverick": "meta-llama/llama-4-maverick",
        # Frontier / proprietary
        "gemini-3.1-flash-lite": "google/gemini-3.1-flash-lite-preview",
        "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
        "gpt-5.2-chat": "openai/gpt-5.2-chat",
    }

    def __init__(
        self,
        model_name: str,
        config: ExecutionConfig,
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        super().__init__(model_name, config)
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.base_url = "https://openrouter.ai/api/v1"

        # Resolve model ID if short name provided
        self.model_id = self.MODEL_IDS.get(model_name, model_name)

    def is_available(self) -> bool:
        """Check if OpenRouter API is available."""
        if not self.api_key:
            import os
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
            self.site_url = self.site_url or os.environ.get("OPENROUTER_SITE_URL", "")
            self.site_name = self.site_name or os.environ.get("OPENROUTER_SITE_NAME", "MetaGlyph")
        return bool(self.api_key)

    def generate(self, prompt: str) -> ExecutionResult:
        """Generate using OpenRouter API."""
        import requests

        import os
        start_time = time.time()
        prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]

        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_id,
                raw_output="",
                success=False,
                error_message="OPENROUTER_API_KEY not set",
                execution_time_ms=0,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "frequency_penalty": self.config.frequency_penalty,
                    "presence_penalty": self.config.presence_penalty,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=self.config.timeout_seconds,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                result = response.json()
                choices = result.get("choices", [])
                output_text = ""
                if choices:
                    message = choices[0].get("message", {})
                    output_text = message.get("content", "")

                usage = result.get("usage", {})

                return ExecutionResult(
                    prompt_id=prompt_id,
                    model=self.model_id,
                    raw_output=output_text,
                    success=True,
                    error_message=None,
                    execution_time_ms=elapsed_ms,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    config=asdict(self.config),
                )
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg}: {error_data['error'].get('message', response.text)}"
                except Exception:
                    error_msg = f"{error_msg}: {response.text}"

                return ExecutionResult(
                    prompt_id=prompt_id,
                    model=self.model_id,
                    raw_output="",
                    success=False,
                    error_message=error_msg,
                    execution_time_ms=elapsed_ms,
                    input_tokens=None,
                    output_tokens=None,
                    config=asdict(self.config),
                )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                prompt_id=prompt_id,
                model=self.model_id,
                raw_output="",
                success=False,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
                input_tokens=None,
                output_tokens=None,
                config=asdict(self.config),
            )

    def get_model_version(self) -> Optional[str]:
        """Return the model ID as version."""
        return self.model_id


class ModelExecutor:
    """Main class for executing prompts against models."""

    BACKENDS = {
        "ollama": OllamaBackend,
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
        "openrouter": OpenRouterBackend,
    }

    # Output format strings per family
    OUTPUT_FORMATS = {
        "1_selection_classification": "Return your answer as a JSON array of strings.",
        "2_structured_extraction": "Return your answer as a JSON object.",
        "3_constraint_composition": "Return your answer as a JSON object with 'selected_ids' and 'count'.",
        "4_conditional_transformation": "Return your answer as a JSON array of transformed objects.",
    }

    def __init__(
        self,
        prompts_dir: Path | str,
        tasks_dir: Path | str,
        outputs_dir: Path | str,
        runs_dir: Path | str,
        model_name: str,
        backend: str = "ollama",
        config: Optional[ExecutionConfig] = None,
    ):
        self.prompts_dir = Path(prompts_dir)
        self.tasks_dir = Path(tasks_dir)
        self.outputs_dir = Path(outputs_dir)
        self.runs_dir = Path(runs_dir)
        self.model_name = model_name
        self.config = config or ExecutionConfig()

        # Initialize backend
        backend_class = self.BACKENDS.get(backend, OllamaBackend)
        self.backend = backend_class(model_name, self.config)

    def check_availability(self) -> bool:
        """Check if the model backend is available."""
        return self.backend.is_available()

    def execute_all(self, skip_existing: bool = True) -> dict[str, list[ExecutionResult]]:
        """Execute all prompts."""
        all_results = {}

        # Get list of families
        families = [d.name for d in self.prompts_dir.iterdir() if d.is_dir()]
        total_families = len(families)

        print(f"\n[{self.model_name}] Starting execution for {total_families} families...")

        prompt_file_count = sum(
            len([
                path for path in list_files(self.prompts_dir / family_name, "*.txt")
                if get_prompt_condition(path.stem) in CONDITIONS
            ])
            for family_name in families
        )
        if prompt_file_count == 0:
            raise RuntimeError(
                f"No Stage 2 prompt files found in {self.prompts_dir}. "
                "Run Stage 2 before model execution."
            )

        for family_idx, family_name in enumerate(sorted(families), 1):
            print(f"\n[{self.model_name}] Family {family_idx}/{total_families}: {family_name}")
            results = self._execute_family(family_name, skip_existing)
            all_results[family_name] = results

            executed = len(results)
            successful = sum(1 for r in results if r.success)
            print(f"[{self.model_name}] Completed: {executed} executed, {successful} successful")

        return all_results

    def execute_family(self, family_name: str, skip_existing: bool = True) -> list[ExecutionResult]:
        """Execute prompts for a specific family."""
        return self._execute_family(family_name, skip_existing)

    def _execute_family(self, family_name: str, skip_existing: bool) -> list[ExecutionResult]:
        """Internal method to execute family prompts.

        Executes the per-instance full prompt files generated by Stage 2. This
        avoids silently diverging from the prompt artifacts that Stage 3
        validates and Stage 5 evaluates.
        """
        prompts_family_dir = self.prompts_dir / family_name
        output_dir = self.outputs_dir / self.model_name / family_name
        runs_dir_model = self.runs_dir / self.model_name / family_name

        ensure_dir(output_dir)
        ensure_dir(runs_dir_model)

        prompt_files = [
            path for path in sorted(list_files(prompts_family_dir, "*.txt"))
            if get_prompt_condition(path.stem) in CONDITIONS
        ]
        if not prompt_files:
            return []

        to_run = []
        skipped = 0
        for prompt_file in prompt_files:
            if skip_existing and (output_dir / f"{prompt_file.stem}.txt").exists():
                skipped += 1
            else:
                to_run.append(prompt_file)

        if skipped > 0:
            print(f"  Skipping {skipped} existing, executing {len(to_run)} remaining...")

        results = []

        def run_one(prompt_file: Path) -> ExecutionResult:
            prompt_id = prompt_file.stem
            full_prompt = load_text(prompt_file)
            result = self._execute_single(prompt_id, full_prompt)
            self._save_result(result, output_dir, runs_dir_model)
            return result

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(run_one, pf) for pf in to_run]
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  [{i}/{len(to_run)}] UNEXPECTED ERROR: {e}")
                    continue
                status = (
                    f"OK ({result.execution_time_ms}ms)"
                    if result.success
                    else f"FAILED: {result.error_message[:50]}"
                )
                print(f"  [{i}/{len(to_run)}] {result.prompt_id}... {status}")
                results.append(result)

        return results

    def _assemble_prompt(self, instruction: str, input_text: str, output_format: str) -> str:
        """Assemble full prompt from instruction, input, and output format."""
        return f"""### Instruction
{instruction}

### Input
{input_text}

### Output Format
{output_format}

### Response"""

    def _execute_single(self, prompt_id: str, prompt_text: str) -> ExecutionResult:
        """Execute a single prompt."""
        result = self.backend.generate(prompt_text)
        result.prompt_id = prompt_id
        return result

    def _save_result(self, result: ExecutionResult, output_dir: Path, runs_dir: Path) -> None:
        """Save execution result and metadata."""
        import datetime

        # Save raw output
        save_text(result.raw_output, output_dir / f"{result.prompt_id}.txt")

        # Create and save run metadata
        prompt_hash = hashlib.md5(result.raw_output.encode()).hexdigest()[:16]

        metadata = RunMetadata(
            prompt_id=result.prompt_id,
            model=result.model,
            model_version=self.backend.get_model_version(),
            timestamp=datetime.datetime.now().isoformat(),
            execution_time_ms=result.execution_time_ms,
            config=result.config,
            prompt_hash=prompt_hash,
            success=result.success,
            error=result.error_message,
        )

        save_json(metadata.to_dict(), runs_dir / f"{result.prompt_id}.meta")

    def execute_prompt(self, prompt_id: str, prompt_text: str) -> ExecutionResult:
        """Execute a single prompt directly."""
        return self._execute_single(prompt_id, prompt_text)


class BatchExecutor:
    """Utility for batch execution with progress tracking."""

    def __init__(self, executor: ModelExecutor):
        self.executor = executor

    def execute_with_progress(
        self,
        prompts: list[tuple[str, str]],  # (prompt_id, prompt_text)
        callback: Optional[callable] = None,
    ) -> list[ExecutionResult]:
        """
        Execute prompts with progress callback.

        callback(current, total, result) is called after each execution.
        """
        results = []
        total = len(prompts)

        for i, (prompt_id, prompt_text) in enumerate(prompts):
            result = self.executor.execute_prompt(prompt_id, prompt_text)
            results.append(result)

            if callback:
                callback(i + 1, total, result)

        return results
