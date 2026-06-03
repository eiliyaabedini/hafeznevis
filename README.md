# hafeznevis — a tiny GPT that writes Hafez (حافظ‌نویس)

A character-level GPT trained from scratch on the **Divan of Hafez** — the collected
poems of the 14th-century Persian poet Hafez of Shiraz. It learns the shape of Persian
ghazals one character at a time and generates new, Hafez-flavored verse.

This is the Persian-poetry counterpart to Andrej Karpathy's
[*Let's build GPT from scratch*](https://www.youtube.com/watch?v=kCc8FmEb1nY) tiny-Shakespeare
demo: a decoder-only Transformer built up piece by piece, here pointed at Hafez instead of
Shakespeare.

```
غزل
الا یا ایها الساقی ادر کاسا و ناولها
که عشق آسان نمود اول ولی افتاد مشکلها
...
```

## What's inside

| File | Description |
|------|-------------|
| `gpt.py` | The full decoder-only Transformer + training loop. Trains, then saves `hafez_model.pt`. |
| `sample.py` | Loads the saved checkpoint and generates poetry instantly (no retraining). |
| `hafez.txt` | The cleaned training corpus — ~275K characters, ~495 ghazals. |
| `Divan_hafez_[www.ketabesabz.com].pdf` | The original source PDF the corpus was extracted from. |

## The data

The corpus was extracted from the source PDF and cleaned for character-level training:

- **Unicode NFKC** normalization — collapses Arabic *presentation-form* glyphs back to
  standard letters (so each letter is one codepoint, not 2–4 positional variants)
- Stripped all RTL/bidi control marks and zero-width characters
- Persian letter normalization (`ي→ی`, `ك→ک`, `ى→ی`)
- Removed page numbers, publisher metadata, and decorative markers
- Verses kept intact and in correct reading order

Result: a clean **44-character vocabulary** (space, digits 0–9, and the 33 Persian letters)
over ~495 ghazals.

## The model

A decoder-only Transformer, following the *Attention Is All You Need* architecture:

- token + positional embeddings
- 6 Transformer blocks, each = multi-head self-attention + feed-forward
- pre-LayerNorm, residual connections, dropout
- ~10.8M parameters

Default hyperparameters (in `gpt.py`):

```
n_embd = 384, n_head = 6, n_layer = 6
block_size = 256, batch_size = 64
learning_rate = 3e-4, max_iters = 5000, dropout = 0.2
```

## Usage

Requires Python 3 and PyTorch (`pip install torch`). Runs on CUDA, Apple-Silicon MPS, or CPU.

**Train once** (produces `hafez_model.pt`):

```bash
python3 gpt.py
```

**Generate anytime** (loads the checkpoint, ~1s):

```bash
python3 sample.py              # 2000 characters
python3 sample.py 5000         # 5000 characters
python3 sample.py 2000 "غزل"   # continue from a prompt
```

## A note on results

With ~275K characters and a 10.8M-parameter model, the network learns the *form* of the
poetry — ghazal structure, real Persian words, rhythm — but not coherent meaning, exactly
like the tiny-Shakespeare demo. It's a learning project, not a poet. The corpus is small
relative to the model, so expect some overfitting; smaller dimensions / more dropout / more
data are the levers to experiment with.

## Credits

- **Architecture & method:** Andrej Karpathy, *Let's build GPT from scratch, in code, spelled out.*
- **Poetry:** Hafez of Shiraz (حافظ شیرازی), public domain.
- **Source text:** Divan of Hafez PDF (diglib.sharif.edu / ketabesabz.com).
