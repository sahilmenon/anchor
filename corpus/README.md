# Corpus

12–15 public filings, hand-labelled. Deliberately ugly.

## What this corpus is a proxy for, and where it falls short

Private-credit origination platforms do not receive clean filings. They receive what
a debt advisor, broker, investment bank or PE sponsor sends: information memoranda,
teasers, credit papers, borrower financial packs, data-room dumps. Often scanned.
Often a spreadsheet exported to PDF by someone in a hurry.

**I do not have access to broker information memoranda.** Nobody outside the industry
does. So this corpus uses public documents chosen to resemble that distribution rather
than the easy end of it:

- **ASIC-lodged financials for small proprietary companies** — thin disclosure, no XBRL, inconsistent line-item naming
- **Charity and NFP financial statements** — genuinely messy, freely public, rarely templated
- **Small council and co-operative financials** — long, badly laid out, tables that span page breaks

**Explicitly avoided: SEC EDGAR 10-Ks.** They are XBRL-tagged, consistently structured
and machine-readable by design. Scoring an extractor on EDGAR measures the wrong thing
and flatters the result.

### Ways the real distribution is harder than this proxy

1. **Scans.** A meaningful share of real submissions are photographed or scanned, so
   text extraction degrades to OCR quality. 3–4 documents here are scanned; in the real
   inbox the proportion is higher.
2. **Adjusted figures with no audit trail.** An IM will state "Adjusted EBITDA" with
   add-backs negotiated between sponsor and lender and justified in a footnote, or in a
   separate spreadsheet, or in an email. Statutory accounts rarely do this.
3. **Forward-looking numbers.** IMs contain projections alongside historicals. Telling
   them apart is a task this corpus cannot test at all.
4. **Deliberate framing.** A teaser is a sales document. Statutory accounts are not.

## Labelling protocol

Each document gets one `corpus/golden/<doc_id>.json` matching `GoldenRecord` in
`src/anchor/schema.py`.

Rules I followed, so that anyone can check the labels:

- **Label only what the document states.** If a figure is not present, the golden value
  is `null`. That is not a gap in the label — it is the correct answer, and an extractor
  that invents a number for it has hallucinated.
- **Flag every judgement call with `ambiguous: true`.** Whenever I had to decide rather
  than read, the field is marked. This set is the abstention test set and it is the most
  interesting part of the corpus.
- **Record a `note`** explaining any ambiguous call, so the label is auditable rather
  than asserted.

### Why `ambiguous` matters more than accuracy

Adjusted EBITDA add-backs are **negotiated, not standard**. DSCR depends on how a given
lender's mandate defines CFADS and debt service. Two competent credit analysts can read
the same document and produce different adjusted EBITDA, and both be defensible.

So the question this harness asks is not only "was the number right" but "did the system
know when not to answer". For a credit decision, a confident wrong EBITDA is worse than
a flagged one.

**DSCR is where this shows up hardest.** A set of statutory financial statements very
often does not disclose principal repayments, so DSCR is not computable from the document
at all. The correct behaviour is abstention. The domain forces the question rather than
the harness manufacturing it.

## Files

- `golden/*.json` — one `GoldenRecord` per document (committed)
- `pdfs/*.pdf` — source documents (**git-ignored**; see `sources.md` for provenance)
- `sources.md` — where each document came from, so the corpus is reproducible
- `thresholds.json` — the regression gate's committed floors

## Honest limits

- **N is 12–15.** Confidence intervals are wide. Every number reported from this corpus
  carries its N, and none of them should be read as a benchmark result.
- **One labeller, no adjudication.** A second labeller would surface disagreements that a
  single pass cannot, and the disagreement rate would itself be a useful statistic.
- **Australian and NFP-weighted.** Not representative of US private-credit deal flow,
  which is where most of the real volume sits.
