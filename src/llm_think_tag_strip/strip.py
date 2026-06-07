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


def _build_opener_pattern(tags: tuple[str, ...], markdown_style: bool) -> re.Pattern[str]:
    """Build one combined regex matching the *opener* of any closed block form.

    Scanning for openers left-to-right (then locating each opener's matching
    closer separately) excises every block in true source order regardless of
    which form it uses, while staying linear in the input size. The previous
    single combined ``<open>.*?</close>`` pattern was quadratic when the input
    held many openers with no closer (e.g. ``"<think>" * n``): a lazy ``.*?``
    re-scanned to end-of-string from every opener position. Splitting opener and
    closer lets us stop the whole pass as soon as one opener has no closer ahead
    (no later opener can have one either), avoiding that blow-up.

    Each alternative names its tag group uniquely (``tag_a``/``tag_p``/``tag_m``)
    because Python's :mod:`re` forbids duplicate group names in one pattern.
    """
    if not tags:
        raise ValueError("tags must be a non-empty tuple")
    alt = "|".join(re.escape(t) for t in tags)
    angle = rf"<(?P<tag_a>{alt})\s*>"
    pipe = rf"<\|(?P<tag_p>{alt})\|>"
    parts = [angle, pipe]
    if markdown_style:
        parts.append(rf"#{{1,6}}[ \t]*(?P<tag_m>{alt})[ \t]*\n")
    combined = "|".join(f"(?:{p})" for p in parts)
    return re.compile(combined, re.IGNORECASE)


def _build_closer(form: str, tag: str) -> re.Pattern[str]:
    """Build the closer regex for a single opener ``form``/``tag`` pair.

    ``form`` is ``"a"`` (angle), ``"p"`` (bracketed-pipe), or ``"m"`` (markdown).
    The markdown closer absorbs the newline that precedes ``### end <tag>`` so it
    is excised with the block, exactly as the old combined pattern did.
    """
    esc = re.escape(tag)
    if form == "a":
        pattern = rf"</{esc}\s*>"
    elif form == "p":
        pattern = rf"</\|{esc}\|>"
    else:  # markdown
        pattern = rf"\n#{{1,6}}[ \t]*end[ \t]+{esc}[ \t]*(?:\n|$)"
    return re.compile(pattern, re.IGNORECASE)


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
        "_opener",
        "_closer_cache",
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
        # One combined opener pattern; for each opener we then locate its own
        # closer. Scanning left-to-right keeps the documented "thinking blocks in
        # source order" contract even when the angle/pipe/markdown forms are
        # interleaved, and stays linear in the input (the old single
        # open-.*?-close pattern was quadratic on many-openers-no-closer input).
        self._opener = _build_opener_pattern(tags, markdown_style)
        # Closers are cheap and per (form, tag); cache the compiled patterns so a
        # reused Stripper does not recompile them on every strip() call.
        self._closer_cache: dict[tuple[str, str], re.Pattern[str]] = {}
        self._unclosed_angle = _build_unclosed_angle_pattern(tags)
        self._unclosed_pipe = _build_unclosed_pipe_pattern(tags)
        if markdown_style:
            self._unclosed_markdown = _build_unclosed_markdown_pattern(tags)
        else:
            self._unclosed_markdown = None

    def _closer(self, form: str, tag: str) -> re.Pattern[str]:
        key = (form, tag)
        pat = self._closer_cache.get(key)
        if pat is None:
            pat = _build_closer(form, tag)
            self._closer_cache[key] = pat
        return pat

    def strip(self, text: str) -> StrippedResult:
        """Strip thinking blocks from ``text`` and return :class:`StrippedResult`."""
        if not text:
            return StrippedResult(clean="", thinking=[], had_thinking=False)

        thinking: list[str] = []

        # Closed blocks: scan openers left-to-right and excise each block up to
        # its matching closer. Captures land in true source order even when forms
        # are interleaved. As soon as an opener has no closer ahead of it we
        # stop: no later opener can have one either, and the remaining text is
        # handed to the unclosed-block phase below.
        kept: list[str] = []
        pos = 0
        while True:
            m = self._opener.search(text, pos)
            if m is None:
                kept.append(text[pos:])
                break
            if m.group("tag_a") is not None:
                form, tag = "a", m.group("tag_a")
            elif m.group("tag_p") is not None:
                form, tag = "p", m.group("tag_p")
            else:
                form, tag = "m", m.group("tag_m")
            close = self._closer(form, tag).search(text, m.end())
            if close is None:
                kept.append(text[pos:])
                break
            kept.append(text[pos : m.start()])
            thinking.append(text[m.end() : close.start()].strip())
            pos = close.end()

        out = "".join(kept)

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
