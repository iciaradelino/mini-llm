# Data

Nothing in `raw/`, `clean/` or `tokenized/` is committed; everything is regenerated
by the scripts in `src/` (see `REPRODUCE.md`).

- `raw/` — downloaded public-domain texts (Project Gutenberg), unmodified
- `clean/` — Gutenberg headers and licence text removed; split into train / val
  (one or two complete works held out for validation)
- `tokenized/` — `train.bin`, `val.bin` (BPE ~8,000 tokens, trained on train only),
  plus a tiny subset (a few MB) for tests

## Sources

| Work | Author | Gutenberg ID / URL | Split |
|------|--------|--------------------|-------|
| TODO |        |                    | train |
| TODO |        |                    | val   |
