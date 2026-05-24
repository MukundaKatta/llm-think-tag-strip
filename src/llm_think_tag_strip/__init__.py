"""llm-think-tag-strip - strip <thinking>/<think> reasoning blocks from LLM output.

Many models emit private reasoning that the end user should not see:

  * Claude with extended thinking emits ``<thinking>...</thinking>`` blocks.
  * DeepSeek-R1 and several open models emit ``<think>...</think>`` blocks.
  * A handful of fine-tunes use ``<|thinking|>...</|thinking|>``.

This library is a tiny zero-dependency function that splits an output string
into a clean answer plus the extracted thinking blocks::

    from llm_think_tag_strip import strip_thinking

    out = strip_thinking("Let me think. <thinking>step1</thinking>The answer is 42.")
    out.clean         # "Let me think. The answer is 42."
    out.thinking      # ["step1"]
    out.had_thinking  # True

The tag set is configurable (defaults cover ``thinking`` and ``think``). An
unclosed open tag is treated as "from open to end of string is thinking" so
truncated streams still return a usable clean answer.
"""

from llm_think_tag_strip.strip import (
    DEFAULT_TAGS,
    StrippedResult,
    Stripper,
    extract_thinking,
    strip_thinking,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_TAGS",
    "Stripper",
    "StrippedResult",
    "__version__",
    "extract_thinking",
    "strip_thinking",
]
