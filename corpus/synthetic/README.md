# Synthetic fixture corpus

Six invented documents, committed as page text, with hand-written labels.

## What this is for, and what it is not

This corpus exists so the regression gate can run on a fresh checkout. The real
corpus (`corpus/`) holds hand-labelled public filings whose PDFs stay
git-ignored, because this repository does not redistribute third-party documents.
That is the right call for the corpus and it leaves CI with nothing to score.

**No number from this corpus is a benchmark result.** These are not filings. Each
was written to contain failure modes chosen in advance, so the baseline's score
here tells you how it handles the traps somebody planted for it and nothing else.
Read the accuracy figure as a tripwire setting.

What it does buy: the whole loop, from parse through extract, verify, score,
derive ratios and gate, runs the same way on any machine. A change that breaks
extraction fails the build instead of shipping.

## Regenerating

```
python corpus/synthetic/build_fixtures.py
```

`build_fixtures.py` holds the document text and the labels together, so you can
check a label against its document in one file. CI regenerates and fails on any
diff, which stops the committed fixtures and the script from drifting apart.

## What each document plants

| Document | Failure mode it was written to provoke |
|---|---|
| `synth-01-acme` | Current and non-current borrowings on separate rows, so a row-anchored read reports one of them as total debt. Expenses in accounting parentheses. |
| `synth-02-harbour` | "Revenue" is a judgement call for an entity whose income is mostly grants. States it has no borrowings, which is a disclosure rather than an absence. A prose mention of EBITDA sits above an unrelated figure. |
| `synth-03-meridian` | Columns run oldest-first, so a row-anchored read takes FY22 where the LTM figure is FY24. Negotiated add-backs. A forward-looking amortisation schedule. |
| `synth-04-riverbend` | OCR damage: `1` read as `l` inside a figure, a space inserted inside "Finance", a corrupted "equivalents". |
| `synth-05-northfield` | Principal split across a borrowings row and a lease row. Debt service printed as outflows, which a naive read returns as negative. |
| `synth-06-kelso` | A segment extract filed ahead of the statutory accounts, so first-hit-wins takes the wrong revenue. A note-reference column between a label and its figure. A statement row wrapped onto two lines. |

## What the baseline does with them

Two cases are worth opening first, because they are what the harness exists to
catch.

**A grounded hallucination.** In `synth-02`, the extractor reports EBITDA of
622,000 and cites a quote that does sit on the page and does contain that number.
Every cheap check passes. The figure is still invented, because the document
never states EBITDA: the label mention and the number belong to different rows.
Grounding is necessary and not sufficient, and this is what that looks like.

**A refused ratio.** In `synth-06`, the facility requires no scheduled principal
amortisation and states that in prose, so the extractor abstains on that field.
The ratio layer then declines to produce a DSCR and names the missing input,
rather than treating the absent leg as zero and reporting coverage of 3.79×.
This is the case the whole project is built around.

**A ratio that stopped refusing.** `synth-05` used to refuse too, for the wrong
reason: interest and principal both came back with the printed sign, debt service
went negative, and the ratio layer bailed out. Normalising debt service to a
positive magnitude fixed that and revealed what the refusal had been hiding. The
extractor reads principal from the borrowings row and misses the lease row
beneath it, so DSCR now computes 1.80× against a true 1.53× and is scored
`disagree`. One bug was masking another, and `ratio.safety` carries a tight floor
in `thresholds.json` to keep the survivor visible.

## Labelling

Same protocol as the real corpus (see `corpus/README.md`): label only what the
document states, `null` where it states nothing, `ambiguous: true` wherever
labelling took a judgement call, and a `note` explaining every call. Eight of the
48 fields here carry the ambiguous flag, and they form the abstention test set.

One difference is worth naming. In the real corpus, you discover ambiguity in
documents somebody else wrote. Here somebody designed it in. That makes the
ambiguous subset useful as a regression check on abstention behaviour and useless
as evidence about how often real documents are ambiguous.
