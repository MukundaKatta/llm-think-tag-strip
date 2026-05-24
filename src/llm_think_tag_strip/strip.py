"""Core thinking-tag stripper."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_TAGS: tuple[str, ...] = ("thinking", "think")
"""Default tag set: Claude extended thinking uses ``thinking``; DeepSeek-R1 and
several open models use ``think``. Pass your own tuple to extend, for example
``("thinking", "think", "reasoning", "reflection", "scratchpad")``."""


@dataclass(frozen=True)
class StrippedResult:
    """Result of stripping thinking tags from a model output string.

    Attributes:
      clean: text with all matched thinking blocks removed and outer whitespace
        trimmed. Double spaces created by the removal are collapsed.
      thinking: extracted thinking blocks in source order. Each entry is the
        inner content of one block, with outer whitespace trimmed.
      had_thinking: True if at least one block was stripped (including an
        unclosed open tag treated as run-to-end-of-string).
    """

    clean: str
    thinking: list[str] = field(default_factory=list)
    had_thinking: bool = False


def _build_angle_pattern(tags: tuple[str, ...]) -> re.Pattern[str]:
    """Build a non-greedy regex matching ``<tag>...</tag>`` for any tag.

    Case-insensitive. ``DOTALL`` so blocks can span newlines.
    """
    if not tags:
        raise ValueError("tags must be a non-empty tuple")
    alt = "|".join(re.escape(t) for t in tags)
    # group(1) is the tag name (used to find the matching closer), group(2) is
    # the inner content. The closer is matched against the same tag name via a
    # named backreference so we don't accept </think> for <thinking>.
    pattern = rf"<(?P<tag>{alt})\s*>(?P<body>.*?)</(?P=tag)\s*>"
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


def _build_pipe_pattern(tags: tuple[str, ...]) -> re.Pattern[str]:
    """Build a regex for the ``<|thinking|>...</|thinking|>`` bracketed-pipe form."""
    alt = "|".join(re.escape(t) for t in tags)
    pattern = rf"<\|(?P<tag>{alt})\|>(?P<body>.*?)</\|(?P=tag)\|>"
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


def _build_markdown_pattern(tags: tuple[str, ...]) -> re.Pattern[str]:
    """Build a regex for the markdown ``### Thinking ... ### End thinking`` form.

    Matches headings of level 1-6. Allows an optional ``end`` marker; if absent,
    runs to end of string (handled by a separate unclosed pattern below).
    """
    alt = "|".join(re.escape(t) for t in tags)
    pattern = (
        rf"#{{1,6}}\s*(?P<tag>{alt})\s*\n"
        rf"(?P<body>.*?)"
        rf"\n#{{1,6}}\s*end\s+(?P=tag)\s*(?:\n|$)"
    )
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


def _build_unclosed_angle_pattern(tags: tuple[str, ...]) -> re.Pattern[str]:
    """Build a regex matching an unclosed ``<tag>`` running to end of string."""
    alt = "|".join(re.escape(t) for t in tags)
    pattern = rf"<(?P<tag>{alt})\s*>(?P<body>.*)\Z"
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


def _build_unclosed_pipe_pattern(tags: tuple[str, ...]) -> re.Pattern[str]:
    alt = "|".join(re.escape(t) for t in tags)
    pattern = rf"<\|(?P<tag>{alt})\|>(?P<body>.*)\Z"
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


def _build_unclosed_markdown_pattern(tags: tuple[str, ...]) -> re.Pattern[str]:
    alt = "|".join(re.escape(t) for t in tags)
    pattern = rf"#{{1,6}}\s*(?P<tag>{alt})\s*\n(?P<body>.*)\Z"
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of two or more spaces/tabs into a single space, then trim.

    Newlines are preserved (only horizontal whitespace is collapsed). This
    matches what users usually want after a mid-line block is excised.
    """
    # Collapse runs of horizontal whitespace (spaces, tabs) to one space.
    collapsed = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse a "space + newline" or "newline + space" introduced by the
    # excised block back to a clean newline.
    collapsed = re.sub(r"[ \t]+\n", "\n", collapsed)
    collapsed = re.sub(r"\n[ \t]+", "\n", collapsed)
    # Collapse 3+ consecutive newlines down to 2 (paragraph break).
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


class Stripper:
    """Reusable stripper with pre-compiled patterns.

    Build once, call ``strip()`` many times. Cheaper than calling the
    module-level :func:`strip_thinking` in a hot loop because the regex
    patterns are not rebuilt on every call.

    Example::

        s = Stripper(tags=("thinking", "think", "reasoning"))
        for chunk in model_outputs:
            print(s.strip(chunk).clean)
    """

    __slots__ = (
        "_tags",
        "_markdown_style",
        "_angle",
        "_pipe",
        "_markdown",
        "_unclosed_angle",
        "_unclosed_pipe",
        "_unclosed_markdown",
    )

    def __init__(
        self,
        tags: tuple[str, ...] = DEFAULT_TAGS,
        markdown_style: bool = False,
    ) -> None:
        if not tags:
            raise ValueError("tags must be a non-empty tuple")
        self._tags = tags
        self._markdown_style = markdown_style
        self._angle = _build_angle_pattern(tags)
        self._pipe = _build_pipe_pattern(tags)
        self._unclosed_angle = _build_unclosed_angle_pattern(tags)
        self._unclosed_pipe = _build_unclosed_pipe_pattern(tags)
        if markdown_style:
            self._markdown = _build_markdown_pattern(tags)
            self._unclosed_markdown = _build_unclosed_markdown_pattern(tags)
        else:
            self._markdown = None
            self._unclosed_markdown = None

    def strip(self, text: str) -> StrippedResult:
        """Strip thinking blocks from ``text`` and return :class:`StrippedResult`."""
        if not text:
            return StrippedResult(clean="", thinking=[], had_thinking=False)

        thinking: list[str] = []
        out = text

        # Closed blocks first, in this order: angle, pipe, markdown (if on).
        # Using a single pass with re.sub + a list-append callback preserves
        # source order while we excise.
        def _capture(m: re.Match[str]) -> str:
            thinking.append(m.group("body").strip())
            return ""

        out = self._angle.sub(_capture, out)
        out = self._pipe.sub(_capture, out)
        if self._markdown is not None:
            out = self._markdown.sub(_capture, out)

        # Unclosed blocks: only match if an open tag exists with no closer left.
        # Try angle, pipe, markdown in that order. Only one unclosed block can
        # match because each consumes through end of string.
        for pat in (self._unclosed_angle, self._unclosed_pipe, self._unclosed_markdown):
            if pat is None:
                continue
            m = pat.search(out)
            if m:
                thinking.append(m.group("body").strip())
                out = out[: m.start()]
                break  # only one unclosed-to-EOF block possible

        clean = _collapse_whitespace(out)
        return StrippedResult(
            clean=clean,
            thinking=thinking,
            had_thinking=bool(thinking),
        )


def strip_thinking(
    text: str,
    tags: tuple[str, ...] = DEFAULT_TAGS,
    markdown_style: bool = False,
) -> StrippedResult:
    """Strip thinking-tag blocks from ``text``.

    Args:
      text: the raw model output.
      tags: tag names to match (case-insensitive). Defaults to ``("thinking", "think")``
        which covers Claude extended thinking and DeepSeek-R1.
      markdown_style: also strip ``### Thinking ... ### End thinking`` blocks.
        Off by default because the ``###`` heading is common in normal answers.

    Returns:
      :class:`StrippedResult` with the cleaned text, the extracted thinking
      bodies (in source order), and a ``had_thinking`` flag.

    Notes:
      * Angle (``<tag>...</tag>``) and bracketed-pipe (``<|tag|>...</|tag|>``)
        forms are always recognized.
      * An unclosed open tag is treated as "from open to end of string is
        thinking". ``clean`` is everything before the open tag.
      * Tag matching is non-greedy, so nested same-tag pairs use the innermost
        closer (standard regex behavior).
    """
    return Stripper(tags=tags, markdown_style=markdown_style).strip(text)


def extract_thinking(
    text: str,
    tags: tuple[str, ...] = DEFAULT_TAGS,
    markdown_style: bool = False,
) -> list[str]:
    """Return just the thinking blocks (in source order) from ``text``.

    Convenience wrapper over :func:`strip_thinking` when you only want the
    thinking content and don't care about the cleaned answer.
    """
    return strip_thinking(text, tags=tags, markdown_style=markdown_style).thinking
