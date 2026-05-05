"""Prompt condition names and parsing helpers."""

from enum import Enum


class PromptCondition(str, Enum):
    """Instruction variants used throughout the pipeline."""

    NL = "NL"
    NL_SHORT = "NL_SHORT"
    ASCII_DSL = "ASCII_DSL"
    MG = "MG"
    CTRL = "CTRL"
    CTRL_RANDOM = "CTRL_RANDOM"


CONDITIONS = [condition.value for condition in PromptCondition]

CONTROL_CONDITIONS = [
    PromptCondition.CTRL.value,
    PromptCondition.CTRL_RANDOM.value,
]


def split_prompt_id(prompt_id: str):
    """Split ``<instance_id>_<condition>`` while allowing underscores in both parts."""
    if prompt_id.endswith("_CTRL_RANDOM"):
        return _split_known_condition(prompt_id, PromptCondition.CTRL_RANDOM)
    if prompt_id.endswith("_ASCII_DSL"):
        return _split_known_condition(prompt_id, PromptCondition.ASCII_DSL)
    if prompt_id.endswith("_NL_SHORT"):
        return _split_known_condition(prompt_id, PromptCondition.NL_SHORT)
    if prompt_id.endswith("_CTRL"):
        return _split_known_condition(prompt_id, PromptCondition.CTRL)
    if prompt_id.endswith("_MG"):
        return _split_known_condition(prompt_id, PromptCondition.MG)
    if prompt_id.endswith("_NL"):
        return _split_known_condition(prompt_id, PromptCondition.NL)

    known_conditions = ", ".join(CONDITIONS)
    raise ValueError(
        f"Prompt id must end with one of [{known_conditions}]: {prompt_id}"
    )


def _split_known_condition(
    prompt_id: str,
    condition: PromptCondition,
) -> tuple[str, str]:
    """Split a prompt id after its condition suffix has already been identified."""
    suffix = f"_{condition.value}"
    instance_id = prompt_id.removesuffix(suffix)
    if not instance_id:
        raise ValueError(
            f"Prompt id is missing an instance id before {suffix}: {prompt_id}"
        )
    return instance_id, condition.value


def get_prompt_condition(prompt_id: str):
    """Return the condition suffix for a prompt id, or None if unknown."""
    try:
        return split_prompt_id(prompt_id)[1]
    except ValueError:
        return None
