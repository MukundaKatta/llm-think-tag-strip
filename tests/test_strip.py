import pytest

from llm_think_tag_strip import (
    DEFAULT_TAGS,
    StrippedResult,
    Stripper,
    extract_thinking,
    strip_thinking,
)

# ---------- single-block basics ----------


def test_single_block_stripped():
    raw = "Let me think. <thinking>step1, step2</thinking>The answer is 42."
    r = strip_thinking(raw)
    assert r.clean == "Let me think. The answer is 42."
    assert r.thinking == ["step1, step2"]
    assert r.had_thinking is True


def test_no_blocks_returns_clean_equal_to_input():
    raw = "No reasoning here at all, just an answer."
    r = strip_thinking(raw)
    assert r.clean == raw
    assert r.thinking == []
    assert r.had_thinking is False


def test_empty_string_returns_empty_clean():
    r = strip_thinking("")
    assert r.clean == ""
    assert r.thinking == []
    assert r.had_thinking is False


def test_default_tags_catch_both_thinking_and_think():
    a = strip_thinking("a <thinking>x</thinking> b")
    b = strip_thinking("a <think>y</think> b")
    assert a.thinking == ["x"]
    assert b.thinking == ["y"]


# ---------- multiple blocks ----------


def test_multiple_blocks_preserved_in_order():
    raw = "A <think>one</think> B <think>two</think> C <thinking>three</thinking> D"
    r = strip_thinking(raw)
    assert r.thinking == ["one", "two", "three"]
    assert r.clean == "A B C D"
    assert r.had_thinking is True


def test_multiple_blocks_same_tag():
    r = strip_thinking("<think>a</think>x<think>b</think>y<think>c</think>")
    assert r.thinking == ["a", "b", "c"]
    assert r.clean == "xy"


def test_only_thinking_text_leaves_empty_clean():
    r = strip_thinking("<thinking>everything is reasoning</thinking>")
    assert r.clean == ""
    assert r.thinking == ["everything is reasoning"]
    assert r.had_thinking is True


# ---------- case insensitivity ----------


def test_case_insensitive_tag_match():
    r = strip_thinking("a <Thinking>X</Thinking> b <THINK>Y</THINK> c")
    assert r.thinking == ["X", "Y"]
    assert r.clean == "a b c"


def test_mismatched_tag_pair_does_not_match():
    # opener <thinking> with closer </think> should NOT pair up.
    r = strip_thinking("a <thinking>oops</think> b")
    # the opener has no proper closer, so the unclosed-path treats it as
    # "from open to end of string" -> body = "oops</think> b".
    assert r.had_thinking is True
    assert r.clean == "a"
    assert r.thinking == ["oops</think> b"]


# ---------- custom tag tuples ----------


def test_custom_tag_tuple():
    r = strip_thinking(
        "Hold on. <reasoning>chain of thought</reasoning>Done.",
        tags=("reasoning",),
    )
    assert r.clean == "Hold on. Done."
    assert r.thinking == ["chain of thought"]


def test_extended_tag_set_with_reflection_and_scratchpad():
    text = "x <reflection>r1</reflection> y <scratchpad>s1</scratchpad> z"
    r = strip_thinking(text, tags=("thinking", "think", "reasoning", "reflection", "scratchpad"))
    assert r.thinking == ["r1", "s1"]
    assert r.clean == "x y z"


def test_empty_tags_tuple_raises():
    with pytest.raises(ValueError):
        strip_thinking("x", tags=())


# ---------- unclosed tag handling ----------


def test_unclosed_tag_runs_to_end_of_string():
    raw = "Answer is 42. <thinking>here is why and then it cuts off"
    r = strip_thinking(raw)
    assert r.clean == "Answer is 42."
    assert r.thinking == ["here is why and then it cuts off"]
    assert r.had_thinking is True


def test_unclosed_only_no_visible_answer():
    r = strip_thinking("<think>just reasoning, nothing else")
    assert r.clean == ""
    assert r.thinking == ["just reasoning, nothing else"]


# ---------- bracketed-pipe form ----------


def test_bracketed_pipe_form():
    raw = "a <|thinking|>private</|thinking|> b"
    r = strip_thinking(raw)
    assert r.thinking == ["private"]
    assert r.clean == "a b"


def test_bracketed_pipe_unclosed():
    r = strip_thinking("answer. <|think|>cut")
    assert r.clean == "answer."
    assert r.thinking == ["cut"]


# ---------- markdown-style opt-in ----------


def test_markdown_style_opt_in():
    md = "Top line.\n### Thinking\nstep one\nstep two\n### End thinking\nBottom line."
    r = strip_thinking(md, markdown_style=True)
    assert "step one" not in r.clean
    assert r.thinking == ["step one\nstep two"]
    assert r.clean.startswith("Top line.")
    assert r.clean.endswith("Bottom line.")


def test_markdown_style_off_by_default():
    md = "Top.\n### Thinking\nstep1\n### End thinking\nBottom."
    r = strip_thinking(md)  # no markdown_style
    # nothing matched, clean keeps the markdown verbatim (whitespace-collapsed)
    assert "### Thinking" in r.clean
    assert r.thinking == []
    assert r.had_thinking is False


def test_markdown_unclosed_runs_to_end():
    md = "Final answer.\n### Thinking\nstep1 then cut"
    r = strip_thinking(md, markdown_style=True)
    assert r.clean == "Final answer."
    assert r.thinking == ["step1 then cut"]


# ---------- whitespace cleanup ----------


def test_whitespace_collapsed_in_clean():
    raw = "before <thinking>middle</thinking>     after"
    r = strip_thinking(raw)
    # the double spaces created by removing the block + the trailing run should
    # collapse to a single space.
    assert r.clean == "before after"


def test_outer_whitespace_trimmed():
    r = strip_thinking("   <think>x</think>   answer   ")
    assert r.clean == "answer"


def test_newlines_inside_block_ok():
    raw = "Hello.\n<thinking>line1\nline2\nline3</thinking>\nWorld."
    r = strip_thinking(raw)
    assert r.thinking == ["line1\nline2\nline3"]
    # the two newlines around the block collapse to a paragraph break
    assert "Hello." in r.clean
    assert "World." in r.clean
    assert "line1" not in r.clean


# ---------- edge cases ----------


def test_thinking_with_special_chars_like_tags_inside():
    # body contains content that looks like another tag but isn't a matching pair
    raw = "x <thinking>note: contains <foo> and <bar/> markup</thinking> y"
    r = strip_thinking(raw)
    assert r.thinking == ["note: contains <foo> and <bar/> markup"]
    assert r.clean == "x y"


def test_very_long_text():
    body = "step " * 5000  # 25_000 chars
    answer = "Answer."
    raw = f"{answer} <thinking>{body}</thinking>"
    r = strip_thinking(raw)
    assert r.clean == "Answer."
    assert r.thinking[0].startswith("step step")
    assert len(r.thinking[0]) >= 24_000


def test_nested_same_tag_uses_innermost_closer():
    # standard non-greedy: <thinking>a<thinking>b</thinking>c</thinking>
    # -> first match consumes "a<thinking>b" and stops at the first </thinking>.
    # the trailing "c</thinking>" then becomes an unmatched closer = leftover text.
    raw = "<thinking>a<thinking>b</thinking>c</thinking>"
    r = strip_thinking(raw)
    assert "b" not in r.thinking or r.thinking[0].endswith("b")
    # we only assert that at least one block was captured and clean does not
    # contain the inner reasoning marker "b".
    assert r.had_thinking is True
    assert "b" not in r.clean


def test_block_with_tag_attribute_like_whitespace():
    # tolerate trailing whitespace inside the opener: <thinking >
    r = strip_thinking("a <thinking >x</thinking> b")
    assert r.thinking == ["x"]
    assert r.clean == "a b"


# ---------- API surface ----------


def test_extract_thinking_convenience():
    out = extract_thinking("<think>a</think>x<think>b</think>")
    assert out == ["a", "b"]


def test_result_is_frozen_dataclass():
    r = strip_thinking("<think>x</think>y")
    assert isinstance(r, StrippedResult)
    with pytest.raises((AttributeError, TypeError)):
        r.clean = "tampered"  # type: ignore[misc]


def test_stripper_reusable_across_calls():
    s = Stripper(tags=("think",))
    # ruff B005 false-positive: these are Stripper.strip(), not str.strip().
    r1 = s.strip("a <think>1</think> b")  # noqa: B005
    r2 = s.strip("c <think>2</think> d")  # noqa: B005
    assert r1.thinking == ["1"]
    assert r2.thinking == ["2"]


def test_stripper_rejects_empty_tags():
    with pytest.raises(ValueError):
        Stripper(tags=())


def test_default_tags_constant_exported():
    assert DEFAULT_TAGS == ("thinking", "think")
