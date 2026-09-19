# Corpus

Anchor scores an extractor against labelled documents. This directory holds two
sets of them, and neither is a benchmark.

| Corpus | Documents | Labels | What it is for |
|---|---|---|---|
| `synthetic/` | 6, invented, committed as page text | All 8 line items, hand-written | Making the regression gate runnable on any checkout |
| `kleister-charity/` | 15 imported from 2,788 available, real | 1 line item, annotated by a third party | Measuring against documents nobody wrote for this project |

`golden/` is empty and stays empty unless you put something in it. It is where
hand labels go if you add them, and the protocol below is how.

## Why two, and why neither is enough alone

`synthetic/` exists because a regression gate with no committed data is a gate
that never runs. Six invented documents, generated from a committed script so
any label can be checked against its document in one file, each planting a named
failure mode that real filings contain. They cannot tell you whether an
extractor works. They can tell you whether it still does what it did last week,
which is the job.

`kleister-charity/` exists because a bespoke corpus produces a number nobody
outside this repository can check.
[Kleister Charity](https://github.com/applicaai/kleister-charity) is 2,788
annual reports filed with the Charity Commission for England and Wales, OCR'd,
long, inconsistently laid out, annotated by someone else, with a published
leaderboard. `anchor import` turns a local checkout into a scoreable corpus.

It annotates one of Anchor's eight line items and no others, so seven are
omitted from every record rather than labelled absent. It also ships no page
separators, so every imported document is one page and grounding figures from it
are not comparable with paginated ones.

Run together they produce the finding this project was built to make possible.
The regex baseline scores 59.4% on the invented documents and **0.0%** on the
real ones, which tells you the synthetic set had been flattering it.

## What this corpus is a proxy for, and where it falls short

Private-credit origination platforms do not receive clean filings. They receive
what a debt advisor, broker, investment bank or PE sponsor sends: information
memoranda, teasers, credit papers, borrower financial packs, data-room dumps.
Often scanned. Often a spreadsheet exported to PDF by someone in a hurry.

Nobody outside the industry has broker information memoranda, so the documents
here are chosen to resemble that distribution rather than the easy end of it.
Charity and council accounts are messy, freely public, and rarely templated.

**Excluded on purpose: SEC EDGAR 10-Ks.** They are XBRL-tagged, consistently
structured and machine-readable by design. Scoring an extractor on EDGAR
measures the wrong thing and flatters the result.

Four ways the real distribution is harder than anything here:

1. **Scans.** A meaningful share of real submissions are photographed, so text
   extraction degrades to OCR quality. The Kleister documents cover this; the
   synthetic ones only simulate it.
2. **Adjusted figures with no audit trail.** An IM states "Adjusted EBITDA" with
   add-backs negotiated between sponsor and lender and justified in a footnote,
   a separate spreadsheet, or an email. Statutory accounts rarely do this.
3. **Forward-looking numbers.** IMs carry projections beside historicals.
   Telling them apart is a task neither corpus tests.
4. **Deliberate framing.** A teaser is a sales document. Statutory accounts are
   not.

## Adding hand labels

Worth doing if you want coverage of the seven line items no external dataset
annotates: EBITDA and its add-backs, total debt, cash, interest expense,
principal repayments, CFADS. That is the gap an import cannot close, and it is
where the interesting judgement calls live.

Each document needs a source file and a label:

```
corpus/pdfs/<doc_id>.pdf        the document   (git-ignored, see below)
corpus/golden/<doc_id>.json     the label      (committed)
```

Drop a PDF into `corpus/pdfs/` and it appears as unlabelled in `anchor serve`,
which extracts it, shows what Anchor read with the page and quote behind every
figure, and lets you record the label field by field. That beats writing the
JSON by hand, and it beats it most on unit scale, which is the mistake that
costs three orders of magnitude.

```bash
anchor serve --corpus corpus     # upload, inspect, label
anchor corpus --corpus corpus    # what is labelled, and does every label resolve
```

Free registers worth pulling from:

| Register | Covers | Notes |
|---|---|---|
| ACNC (`acnc.gov.au`) | Australian charities and NFPs | Annual financial reports on each charity's record |
| UK Charity Commission | England and Wales charities | Full accounts as filed, often scanned. The source Kleister draws from |
| UK Companies House | UK small companies | Thin, inconsistently named accounts, close to the ASIC shape, and free, which ASIC's own document service is not |
| State and local government sites | Councils, co-operatives | Long and badly laid out, which is the point |

Record provenance in `sources.md` as you go. A corpus nobody else can reassemble
is a corpus nobody else can check.

### Why the PDFs are git-ignored and the labels are not

The labels are the work: small, reviewable, and the thing two people need to
agree on. Sharing them through git turns a disagreement into a diff. The
documents behind them are third-party filings this repository has no business
redistributing, so `pdfs/` stays local while `sources.md` carries the provenance
that lets someone else fetch the same files.

## Labelling protocol

Each label is one `corpus/golden/<doc_id>.json` matching `GoldenRecord` in
`src/anchor/schema.py`. Three rules, and the reason for each:

- **Label only what the document states.** If a figure is absent, the golden
  value is `null`. That is the correct answer rather than a gap in the label,
  and an extractor that invents a number for it has hallucinated.
- **Flag every judgement call with `ambiguous: true`.** Whenever you decide
  rather than read, mark the field. This set is the abstention test set and the
  most interesting part of any corpus here.
- **Record a `note`** explaining any ambiguous call, so the label is auditable
  rather than asserted.

One convention catches people out: `interest_expense` and `principal_repayments`
are stored as **positive magnitudes**. A statement prints them parenthesised
because the cash went out, not because the quantity is negative. See
`MAGNITUDE_ITEMS` in `src/anchor/schema.py`.

### Why `ambiguous` matters more than accuracy

Adjusted EBITDA add-backs are **negotiated, not standard**. DSCR depends on how
a given lender's mandate defines CFADS and debt service. Two competent credit
analysts can read the same document and produce different adjusted EBITDA, and
both be defensible.

So the question this harness asks is not only "was the number right" but "did
the system know when not to answer". For a credit decision, a confident wrong
EBITDA is worse than a flagged one.

**DSCR is where this shows up hardest.** A set of statutory financial statements
very often does not disclose principal repayments, so DSCR is not computable
from the document at all. The correct behaviour is abstention. The domain forces
the question rather than the harness manufacturing it.

No external dataset labels either absence or ambiguity. Every one surveyed
annotates values that are present. Kleister is the partial exception, because
its task definition includes decoy keys for which no value should be given, and
those import as true absences.

## Files

- `synthetic/`: six invented documents and their labels (committed)
- `kleister-charity/`: generated by `anchor import` (git-ignored; the source
  dataset declares no licence, so labels derived from it stay local)
- `golden/*.json`: hand labels, if you add them (committed)
- `pdfs/*.pdf`: source documents (git-ignored; see `sources.md`)
- `sources.md`: provenance, so any corpus here can be reassembled
- `thresholds.json`: gate floors. **Provisional**, copied from the synthetic
  corpus, produced by no real run. Re-derive them from a baseline before reading
  a pass here as evidence of anything.

## Limits

- **Every corpus here is small.** Six invented documents and fifteen imported
  ones. Confidence intervals are wide, single documents move accuracy by several
  points, and the ordering between two close extractors is not established. Each
  figure carries its N for that reason.
- **One labeller, no adjudication** on the synthetic set. A second would surface
  disagreements a single pass cannot, and the disagreement rate would itself be
  informative.
- **The imported corpus measures one field** and cannot see the seven where the
  judgement calls live.
- **Geographically narrow.** UK charities and invented Australian companies.
  Neither resembles US private-credit deal flow, where most of the volume sits.
