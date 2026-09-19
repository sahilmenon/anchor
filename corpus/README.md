# Corpus

> **Status: specified, not populated.** `golden/` is empty. No real filing has
> been labelled yet, and no number anywhere in this repository comes from one.
> The 59.4% accuracy, the taxonomy and the gate floors all come from
> `corpus/synthetic/`, six invented documents described in its own README. This
> file sets out what the real corpus is for, how to populate it, and the
> protocol its labels must follow. Read it as a specification rather than a
> description.

## What this corpus is a proxy for, and where it falls short

Private-credit origination platforms do not receive clean filings. They receive
what a debt advisor, broker, investment bank or PE sponsor sends: information
memoranda, teasers, credit papers, borrower financial packs, data-room dumps.
Often scanned. Often a spreadsheet exported to PDF by someone in a hurry.

**Nobody outside the industry has broker information memoranda.** So this corpus
is specified around public documents chosen to resemble that distribution rather
than the easy end of it:

- **ASIC-lodged financials for small proprietary companies**: thin disclosure,
  no XBRL, inconsistent line-item naming
- **Charity and NFP financial statements**: messy, free to obtain, rarely
  templated
- **Small council and co-operative financials**: long, badly laid out, tables
  that span page breaks

**Deliberately excluded: SEC EDGAR 10-Ks.** They are XBRL-tagged, consistently
structured and machine-readable by design. Scoring an extractor on EDGAR
measures the wrong thing and flatters the result.

### Ways the real distribution is harder than this proxy

1. **Scans.** A meaningful share of real submissions are photographed or
   scanned, so text extraction degrades to OCR quality. The target mix here is
   3–4 scanned documents out of 12–15; in the real inbox the proportion is
   higher.
2. **Adjusted figures with no audit trail.** An IM states "Adjusted EBITDA" with
   add-backs negotiated between sponsor and lender and justified in a footnote,
   or in a separate spreadsheet, or in an email. Statutory accounts rarely do
   this.
3. **Forward-looking numbers.** IMs carry projections alongside historicals.
   Telling them apart is a task this corpus cannot test at all.
4. **Deliberate framing.** A teaser is a sales document. Statutory accounts are
   not.

## Populating it

Each document needs two things: the source file, and a label.

```
corpus/pdfs/<doc_id>.pdf        the document   (git-ignored, see below)
corpus/golden/<doc_id>.json     the label      (committed)
```

Drop a PDF into `corpus/pdfs/` and it shows up as unlabelled in `anchor serve`,
which will extract it, show you what Anchor read with the page and quote behind
every figure, and let you record the label field by field. That is faster than
writing the JSON by hand and less error-prone about unit scale, which is the
mistake that matters most.

```bash
anchor serve --corpus corpus     # upload, inspect, label
anchor corpus --corpus corpus    # what is labelled, and does every label resolve
```

Where to get documents, all free and public:

| Register | Covers | Notes |
|---|---|---|
| ACNC charity register (`acnc.gov.au`) | Australian charities and NFPs | Annual financial reports attached to each charity's record. The best free source of messy statements. |
| UK Charity Commission (`register-of-charities.charitycommission.gov.uk`) | England and Wales charities | Full accounts as filed, often scanned. |
| UK Companies House (`find-and-update.company-information.service.gov.uk`) | UK small companies | Small-company accounts are thin and inconsistently named, close to the ASIC shape, and free. ASIC's own document service is not. |
| State and local government sites | Councils, co-operatives | Annual reports; long and badly laid out, which is the point. |

Record every document's provenance in `sources.md` as you add it. A corpus
nobody else can reassemble is a corpus nobody else can check.

### Why the PDFs are git-ignored and the labels are not

The labels are the work: small, reviewable, and the thing two people need to
agree on. Sharing them through git turns a disagreement into a diff. The
documents behind them are third-party filings, and this repository has no
business redistributing them, so `pdfs/` stays local while `sources.md` carries
the provenance that lets someone else fetch the same files.

## Labelling protocol

Each document gets one `corpus/golden/<doc_id>.json` matching `GoldenRecord` in
`src/anchor/schema.py`. Three rules, and the reason for each:

- **Label only what the document states.** If a figure is absent, the golden
  value is `null`. That is the correct answer rather than a gap in the label, and
  an extractor that invents a number for it has hallucinated.
- **Flag every judgement call with `ambiguous: true`.** Whenever you decide
  rather than read, mark the field. This set is the abstention test set and it
  is the most interesting part of the corpus.
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

## Files

- `golden/*.json`: one `GoldenRecord` per document (committed; **currently empty**)
- `pdfs/*.pdf`: source documents (**git-ignored**; see `sources.md` for provenance)
- `sources.md`: where each document came from, so the corpus is reproducible
- `thresholds.json`: the regression gate's committed floors. The bounds in it
  are **provisional**: they were copied from the synthetic corpus and no real
  run has produced them. Re-derive every one from a baseline run before reading
  a pass here as evidence of anything.

## Limits this corpus will still have once it is populated

These are properties of the design, not of its current emptiness.

- **N is 12–15.** Confidence intervals are wide. Every number reported from this
  corpus must carry its N, and none of them should be read as a benchmark
  result.
- **One labeller, no adjudication.** A second labeller would surface
  disagreements a single pass cannot, and the disagreement rate would itself be
  a useful statistic.
- **Australian and NFP-weighted.** Not representative of US private-credit deal
  flow, which is where most of the real volume sits.
