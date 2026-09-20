# Results

Run artefacts behind the figures in the root `README.md`. `runs/` is git-ignored
because a working directory fills with runs nobody will read again; these two
are the ones a reader needs to check a claim, so they are copied here by hand.

| File | Corpus | Extractor | Run |
|---|---|---|---|
| `synthetic-claude-code.json` | `corpus/synthetic`, 6 invented documents | `claude-code` | 2026-09-20 |
| `kleister-charity-claude-code.json` | `corpus/kleister-charity`, 15 real filings | `claude-code` | 2026-09-20 |

Each file holds the full report: taxonomy counts, per-line-item rates, derived
ratio agreement, and the selective-prediction summary. Every figure in the
root README's model row can be read off one of them.

## What `claude-code` is, and what it cannot report

The extractor reaches Claude through the Claude Code CLI's headless mode rather
than the API, because a subscription pays for one and prepaid credits pay for
the other. It inherits the prompt, schema, coercion, quote check and
retry-and-abstain from the API extractor, so a row produced here differs in how
the bytes travelled and nothing Anchor scores.

Two fields are blank and stay blank. `total_cost_usd` is 0.0 and both token
counts are 0 because the CLI reports no usage, so **do not read that zero as a
price**. `extractor` says `claude-code` rather than a model id because the CLI
uses whatever the local install is configured with, and naming a model would
assert something the run cannot check.

Latency is real. It is wall-clock for a process launch plus a completion, which
is slower than an API call and not comparable with one.

## Why the Kleister file has no per-document records

[Kleister Charity](https://github.com/applicaai/kleister-charity) declares no
licence. The per-document records carry quoted passages from the filings, so
publishing them would redistribute the dataset. The `documents` key is withheld
and the aggregate report is not, because counts are not the corpus.

Rebuild the full artefact from a local checkout:

```bash
anchor import kleister-charity --from ./kleister-charity --limit 15
anchor run --corpus corpus/kleister-charity --extractor claude-code
```

Importing takes the first 15 documents in the dataset's own order, so the same
command reproduces the same corpus. The model row will not reproduce exactly:
the CLI answers from whatever model the local install points at, and sampling
varies between runs.
