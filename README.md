# llm-think-tag-strip

[![PyPI](https://img.shields.io/pypi/v/llm-think-tag-strip.svg)](https://pypi.org/project/llm-think-tag-strip/)
[![Python](https://img.shields.io/pypi/pyversions/llm-think-tag-strip.svg)](https://pypi.org/project/llm-think-tag-strip/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Strip `<thinking>` / `<think>` reasoning blocks from LLM output.**

Many LLMs emit private reasoning the end user should not see. Claude extended
thinking returns `<thinking>...</thinking>` blocks. DeepSeek-R1 and several
open models return `<think>...</think>`. This library splits the raw output
into a clean answer plus the extracted thinking, in source order.

Zero runtime deps (stdlib `re` only).

## Install

```bash
pip install llm-think-tag-strip
```

## Use

```python
from llm_think_tag_strip import strip_thinking

raw = "Let me think. <thinking>step1, step2</thinking>The answer is 42."
result = strip_thinking(raw)

result.clean         # "Let me think. The answer is 42."
result.thinking      # ["step1, step2"]
result.had_thinking  # True
```

Multiple blocks are preserved in order:

```python
raw = "A <think>one</think> B <think>two</think> C"
strip_thinking(raw).thinking
# ["one", "two"]
```

Custom tag set:

```python
strip_thinking(
    "Hold on. <reasoning>...</reasoning>Done.",
    tags=("thinking", "think", "reasoning", "reflection", "scratchpad"),
)
```

Markdown-style blocks (opt-in, because `###` headings are common in normal
answers):

```python
md = "Top.\n### Thinking\nstep1\n### End thinking\nDone."
strip_thinking(md, markdown_style=True).clean
# "Top.\nDone."
```

Bracketed-pipe form (`<|thinking|>...</|thinking|>`) is always recognized:

```python
strip_thinking("a <|think|>b</|think|> c").clean
# "a c"
```

Unclosed open tag is treated as "thinking ran to end of stream", which is what
you want for truncated streaming output:

```python
strip_thinking("Answer. <thinking>oops cut off")
# StrippedResult(clean="Answer.", thinking=["oops cut off"], had_thinking=True)
```

Just the extracted blocks:

```python
from llm_think_tag_strip import extract_thinking
extract_thinking("<think>a</think>x<think>b</think>")
# ["a", "b"]
```

Batch use (reuse compiled patterns):

```python
from llm_think_tag_strip import Stripper
s = Stripper(tags=("thinking", "think", "reasoning"), markdown_style=True)
for chunk in many_outputs:
    print(s.strip(chunk).clean)
```

## What it does NOT do

- No streaming parser. Feed completed strings. For incremental streams, buffer
  until you see the closing tag (or end of stream) and pass the buffer in.
- No HTML safety. The output is whatever your model produced minus the
  thinking blocks. Escape it yourself before rendering to HTML.
- No prompt rewriting. This only operates on model output, not on the input.

## Siblings

Same author, same agent-stack family:

- [`llm-output-validator`](https://pypi.org/project/llm-output-validator/) -
  validate the `clean` part against a schema.
- [`tool-output-truncate-py`](https://pypi.org/project/tool-output-truncate-py/) -
  truncate tool output before sending it back to the model.
- [`llm-prompt-compress`](https://pypi.org/project/llm-prompt-compress/) -
  shrink prompts before they hit the model.

## License

MIT
