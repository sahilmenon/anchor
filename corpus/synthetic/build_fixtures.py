"""Build the synthetic fixture corpus.

Why a synthetic corpus exists at all
------------------------------------
The real corpus (`corpus/golden`, `corpus/pdfs`) is hand-labelled public
filings, and the PDFs are git-ignored -- they are third-party documents and
this repository does not redistribute them. That is the right call for the
corpus and the wrong one for CI: a regression gate with no committed data is a
gate that never runs.

So this file writes a small set of *invented* documents, committed as page
text, with hand-written golden labels. They are not filings, they are not
sampled from anything, and they produce no benchmark number. What they do is
exercise the whole loop -- parse, extract, verify, score, derive ratios, gate --
deterministically on any checkout, so a change that quietly breaks extraction
cannot merge.

Every document here was written to contain a *specific, named* failure mode
that real credit documents contain; `corpus/synthetic/README.md` lists them.
The labels follow the same protocol as the real corpus: label only what the
document states, mark judgement calls `ambiguous`, and record why.

Run: python corpus/synthetic/build_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ===========================================================================
# synth-01 -- a clean small-proprietary statement set.
# Planted: current and non-current borrowings on separate rows, which a
# row-anchored extractor cannot add together.
# ===========================================================================
S01_P1 = """\
ACME MANUFACTURING PTY LTD
ABN 12 345 678 901
Financial Report for the year ended 30 June 2024

STATEMENT OF PROFIT OR LOSS
For the year ended 30 June 2024
All amounts in A$'000

                                              2024        2023
Total revenue                               24,180      21,405
Cost of sales                              (14,902)    (13,388)
Gross profit                                 9,278       8,017
Employee benefits expense                   (4,113)     (3,902)
Depreciation and amortisation               (1,204)     (1,118)
Other operating expenses                    (1,214)     (1,017)
Finance costs                                 (842)       (774)
Profit before income tax                     1,905       1,406
Income tax expense                            (572)       (422)
Profit for the year                          1,333         984
"""

S01_P2 = """\
ACME MANUFACTURING PTY LTD

STATEMENT OF FINANCIAL POSITION
As at 30 June 2024
All amounts in A$'000

                                              2024        2023
Current assets
Cash and cash equivalents                    1,842       1,196
Trade and other receivables                  3,905       3,412
Inventories                                  4,118       3,880
Total current assets                         9,865       8,488

Non-current assets
Property, plant and equipment               12,404      11,908
Total assets                                22,269      20,396

Current liabilities
Trade and other payables                     3,112       2,905
Borrowings                                   1,500       1,500
Total current liabilities                    4,612       4,405

Non-current liabilities
Borrowings                                   8,750      10,250
Total non-current liabilities                8,750      10,250

Total liabilities                           13,362      14,655
Net assets                                   8,907       5,741
"""

S01_P3 = """\
ACME MANUFACTURING PTY LTD

STATEMENT OF CASH FLOWS
For the year ended 30 June 2024
All amounts in A$'000

                                              2024        2023
Cash flows from operating activities
Receipts from customers                     25,910      22,804
Payments to suppliers and employees        (21,455)    (19,102)
Interest paid                                 (838)       (770)
Income taxes paid                             (498)       (402)
Net cash from operating activities           3,119       2,530

Cash flows from financing activities
Repayment of borrowings                     (1,500)     (1,500)
Dividends paid                                (400)       (300)
Net cash used in financing activities       (1,900)     (1,800)
"""


# ===========================================================================
# synth-02 -- a not-for-profit special purpose report.
# Planted: "revenue" is genuinely ambiguous for an entity whose income is
# mostly grants; the entity states it has no borrowings (a fact, not an
# absence); and a prose mention of EBITDA sits directly above an unrelated
# figure, which is how a label-anchored read invents a number.
# ===========================================================================
S02_P1 = """\
HARBOUR COMMUNITY SERVICES INCORPORATED
Special Purpose Financial Report
Year ended 31 December 2024
Amounts are stated in thousands of dollars

INCOME AND EXPENDITURE STATEMENT

                                              2024        2023
Turnover                                     3,418       3,102
Grant income                                 2,905       2,740
Fundraising income                             513         362
Total income                                 6,836       6,204
Employee costs                              (4,902)     (4,518)
Occupancy costs                               (612)       (588)
Depreciation                                  (188)       (171)
Surplus for the year                           134          97
"""

S02_P2 = """\
HARBOUR COMMUNITY SERVICES INCORPORATED

STATEMENT OF FINANCIAL POSITION
As at 31 December 2024
Amounts in $'000

                                              2024        2023
Cash at bank                                   918         764
Term deposits                                1,200       1,200
Receivables                                    342         288
Total assets                                 2,460       2,252
Payables and accruals                          405         371
Employee provisions                            622         589
Total liabilities                            1,027         960
Net assets                                   1,433       1,292

The Association has no borrowings and no finance facilities.

EBITDA is not a measure used by the Association.
Note 4 -- Employee provisions                  622         589
"""


# ===========================================================================
# synth-03 -- a sponsor information memorandum.
# Planted: historical columns run oldest-first, so a row-anchored read takes
# FY22 where the LTM figure is FY24. Also carries negotiated add-backs and a
# forward-looking amortisation schedule.
# ===========================================================================
S03_P1 = """\
PROJECT MERIDIAN
Confidential Information Memorandum
Prepared for prospective senior lenders

FINANCIAL SUMMARY
A$ in millions unless otherwise stated

                                        FY22      FY23      FY24
Total revenue                           88.4      96.2     104.7
Reported EBITDA                         11.2      12.8      14.1
Normalisation adjustments                1.9       2.2       2.6
Adjusted EBITDA                         13.1      15.0      16.7
EBITDA margin                           14.8%     15.6%     15.9%
"""

S03_P2 = """\
PROJECT MERIDIAN -- CAPITAL STRUCTURE
As at 30 June 2024, A$ in millions

Senior term loan A                                        42.0
Senior term loan B                                        18.5
Total debt                                                60.5
Cash and cash equivalents                                  6.3
Net debt                                                  54.2

Pro forma leverage (net debt / adjusted EBITDA)            3.2x
Interest expense (LTM)                                     4.8
"""

S03_P3 = """\
PROJECT MERIDIAN -- DEBT SERVICE AND AMORTISATION
A$ in millions

Term loan A amortises at 5.0% per annum of the original facility limit.

Scheduled principal amortisation (FY25)                    2.1
Cash flow available for debt service (LTM)                16.2
"""


# ===========================================================================
# synth-04 -- a scanned co-operative report, OCR text layer.
# Planted: "1" read as "l" inside a figure, a space inserted inside "Finance",
# and a corrupted "equivalents". The damaged revenue figure is the interesting
# one: the number the extractor reports cannot be grounded in the quote it
# cites, so the harness rejects it rather than scoring it as merely wrong.
# ===========================================================================
S04_P1 = """\
RIVERBEND CO-OPERATIVE LIMITED
ANNUAL FlNANClAL REPORT 2024
(scanned document -- text layer produced by OCR)

STATEMENT OF PROFIT OR LOSS
$ '000

Total  revenue                            l2,405
Cost of goods sold                       (8,112)
Gross  profit                             4,293
Fi nance costs                             (391)
Depreciation and amortisatlon              (602)
Profit before tax                          1,105
"""

S04_P2 = """\
RIVERBEND CO-OPERATIVE LIMITED

BALANCE SHEET
$ '000

Cash and cash equlvalents                    287
Trade receivables                          1,904
Total loans and borrowings                 3,850
Members funds                              2,014
"""


# ===========================================================================
# synth-05 -- a statement set where DSCR is computable from the document.
# Planted: expenses printed in accounting parentheses, which a row-anchored
# read returns as negative debt service; and principal split across a
# borrowings row and a lease row.
# ===========================================================================
S05_P1 = """\
NORTHFIELD LOGISTICS PTY LTD
Financial Statements for the year ended 30 June 2024

STATEMENT OF PROFIT OR LOSS
All figures in A$'000

                                              2024
Sales revenue                               41,660
EBITDA                                       6,204
Depreciation and amortisation               (2,118)
Interest expense                            (1,043)
Profit before tax                            3,043
"""

S05_P2 = """\
NORTHFIELD LOGISTICS PTY LTD

STATEMENT OF FINANCIAL POSITION
All figures in A$'000

                                              2024
Cash and cash equivalents                    2,940
Trade receivables                            5,118
Property, plant and equipment               21,405
Total assets                                29,463

Trade payables                               4,012
Bank loan -- secured                        14,200
Lease liabilities                            3,180
Total borrowings                            17,380
Net assets                                   8,071
"""

S05_P3 = """\
NORTHFIELD LOGISTICS PTY LTD

STATEMENT OF CASH FLOWS
All figures in A$'000

                                              2024
Net cash from operating activities           5,880
Interest paid                               (1,043)
Repayment of borrowings                     (2,400)
Repayment of lease liabilities                (610)
Net cash used in financing activities       (3,010)
"""


# ===========================================================================
# synth-06 -- a data-room dump: a segment extract filed ahead of the statutory
# accounts. Planted: first-hit-wins takes the segment revenue instead of the
# consolidated figure; a note-reference column sits between a label and its
# figure; and a statement row wraps onto two lines.
# ===========================================================================
S06_P1 = """\
KELSO GROUP HOLDINGS PTY LTD
Board pack extract -- segment performance
$'000

SEGMENT NOTE -- RETAIL DIVISION ONLY

                                                    2024
Revenue                                            8,220
Segment contribution                               1,014
"""

S06_P2 = """\
KELSO GROUP HOLDINGS PTY LTD

CONSOLIDATED STATEMENT OF PROFIT OR LOSS
For the year ended 30 June 2024
$'000

                                           Note      2024
Revenue from contracts with customers         3    31,905
Earnings before interest tax depreciation and amortisation
                                                   4,882
Finance costs                                 6     1,288
Profit before income tax                           1,905
"""

S06_P3 = """\
KELSO GROUP HOLDINGS PTY LTD

NOTE 14 -- BORROWING FACILITIES
$'000

                                                    2024
Bank overdraft                                       420
Commercial bills                                   6,100
Total interest bearing liabilities                 6,520

NOTE 15 -- CASH
Cash and cash equivalents                          1,105

The Group facility agreement does not require scheduled principal amortisation during the term.
"""


def _f(name, value, page=None, ambiguous=False, note=""):
    return {
        "name": name,
        "value": value,
        "page": page,
        "ambiguous": ambiguous,
        "note": note,
    }


DOCS = [
    {
        "doc_id": "synth-01-acme",
        "source": "Synthetic. Written for this repository; not a real filing.",
        "scanned": False,
        "pages": [S01_P1, S01_P2, S01_P3],
        "fields": [
            _f("revenue_ltm", 24_180_000.0, 1),
            _f(
                "ebitda_reported", None, None, True,
                "Not stated. Derivable as PBT 1,905 + finance costs 842 + D&A 1,204 = 3,951, "
                "but the corpus labels what the document states, and a derived figure is a "
                "judgement about which add-backs belong.",
            ),
            _f("ebitda_addbacks", None, None, False, "Not stated."),
            _f(
                "total_debt", 10_250_000.0, 2,
                note="Current borrowings 1,500 plus non-current borrowings 8,750. Both rows "
                     "are stated; the sum is arithmetic, not judgement.",
            ),
            _f("cash", 1_842_000.0, 2),
            _f(
                "interest_expense", 842_000.0, 1,
                note="Printed as (842). The parentheses are an expense convention, not a "
                     "negative quantity: debt service of 842 is what the statement discloses.",
            ),
            _f(
                "principal_repayments", 1_500_000.0, 3,
                note="Repayment of borrowings 1,500, printed as an outflow.",
            ),
            _f(
                "cfads", None, None, True,
                "Not disclosed. Operating cash flow of 3,119 is the nearest figure, but CFADS "
                "depends on a lender definition of maintenance capex and tax.",
            ),
        ],
    },
    {
        "doc_id": "synth-02-harbour",
        "source": "Synthetic. Written for this repository; not a real filing.",
        "scanned": False,
        "pages": [S02_P1, S02_P2],
        "fields": [
            _f(
                "revenue_ltm", 6_836_000.0, 1, True,
                "Total income 6,836 includes grant income of 2,905. Whether grant income counts "
                "as revenue for a credit assessment is a lender call: turnover alone is 3,418.",
            ),
            _f(
                "ebitda_reported", None, None, False,
                "Not stated. The document says explicitly that it does not use the measure.",
            ),
            _f("ebitda_addbacks", None, None, False, "Not stated."),
            _f(
                "total_debt", 0.0, 2,
                note="The document states the Association has no borrowings and no finance "
                     "facilities. Zero is a disclosure here, not an absence.",
            ),
            _f(
                "cash", 918_000.0, 2, True,
                "Cash at bank 918. Term deposits of 1,200 may or may not be cash equivalents "
                "depending on their term, which the document does not give.",
            ),
            _f("interest_expense", None, None, False, "Not stated."),
            _f("principal_repayments", None, None, False, "Not stated."),
            _f("cfads", None, None, False, "Not stated."),
        ],
    },
    {
        "doc_id": "synth-03-meridian",
        "source": "Synthetic. Written for this repository; not a real filing.",
        "scanned": False,
        "pages": [S03_P1, S03_P2, S03_P3],
        "fields": [
            _f(
                "revenue_ltm", 104_700_000.0, 1,
                note="FY24 column. FY22 and FY23 comparatives are printed to its left.",
            ),
            _f("ebitda_reported", 14_100_000.0, 1, note="FY24 reported EBITDA."),
            _f(
                "ebitda_addbacks", 2_600_000.0, 1, True,
                "Stated as normalisation adjustments of 2.6 in FY24. The memorandum does not "
                "itemise them, so a lender cannot test whether they are recurring.",
            ),
            _f("total_debt", 60_500_000.0, 2),
            _f("cash", 6_300_000.0, 2),
            _f("interest_expense", 4_800_000.0, 2),
            _f(
                "principal_repayments", 2_100_000.0, 3, True,
                "The 2.1 figure is the FY25 scheduled amortisation -- forward-looking, not an "
                "LTM historical. Treating it as debt service is defensible for a DSCR the "
                "facility must carry, but it is not what the other line items measure.",
            ),
            _f("cfads", 16_200_000.0, 3),
        ],
    },
    {
        "doc_id": "synth-04-riverbend",
        "source": "Synthetic. Written for this repository; not a real filing.",
        "scanned": True,
        "pages": [S04_P1, S04_P2],
        "fields": [
            _f(
                "revenue_ltm", 12_405_000.0, 1,
                note="Printed as l2,405 -- the OCR layer has read the leading 1 as a lower-case "
                     "L. A human reads 12,405 from the column arithmetic.",
            ),
            _f("ebitda_reported", None, None, False, "Not stated."),
            _f("ebitda_addbacks", None, None, False, "Not stated."),
            _f("total_debt", 3_850_000.0, 2),
            _f("cash", 287_000.0, 2, note="Printed as equlvalents -- OCR damage."),
            _f(
                "interest_expense", 391_000.0, 1,
                note="Printed as Fi nance costs (391) -- OCR has split the word.",
            ),
            _f("principal_repayments", None, None, False, "Not stated."),
            _f("cfads", None, None, False, "Not stated."),
        ],
    },
    {
        "doc_id": "synth-05-northfield",
        "source": "Synthetic. Written for this repository; not a real filing.",
        "scanned": False,
        "pages": [S05_P1, S05_P2, S05_P3],
        "fields": [
            _f("revenue_ltm", 41_660_000.0, 1),
            _f("ebitda_reported", 6_204_000.0, 1),
            _f("ebitda_addbacks", None, None, False, "Not stated."),
            _f("total_debt", 17_380_000.0, 2),
            _f("cash", 2_940_000.0, 2),
            _f(
                "interest_expense", 1_043_000.0, 1,
                note="Printed as (1,043); see synth-01 on the parentheses convention.",
            ),
            _f(
                "principal_repayments", 3_010_000.0, 3,
                note="Repayment of borrowings 2,400 plus repayment of lease liabilities 610. "
                     "Both are principal under a standard debt-service definition.",
            ),
            _f(
                "cfads", None, None, True,
                "Not disclosed. Operating cash flow of 5,880 is before capex and is not CFADS.",
            ),
        ],
    },
    {
        "doc_id": "synth-06-kelso",
        "source": "Synthetic. Written for this repository; not a real filing.",
        "scanned": False,
        "pages": [S06_P1, S06_P2, S06_P3],
        "fields": [
            _f(
                "revenue_ltm", 31_905_000.0, 2,
                note="Consolidated revenue. The segment extract on page 1 reports 8,220 for the "
                     "retail division only.",
            ),
            _f(
                "ebitda_reported", 4_882_000.0, 2,
                note="The label wraps onto its own line; the figure sits on the line below.",
            ),
            _f("ebitda_addbacks", None, None, False, "Not stated."),
            _f("total_debt", 6_520_000.0, 3),
            _f("cash", 1_105_000.0, 3),
            _f(
                "interest_expense", 1_288_000.0, 2,
                note="Note reference 6 sits between the label and the figure.",
            ),
            _f(
                "principal_repayments", 0.0, 3, True,
                "The facility requires no scheduled principal amortisation during the term, so "
                "scheduled principal is zero. A lender modelling a refinancing at maturity "
                "would not treat debt service as interest-only.",
            ),
            _f("cfads", None, None, False, "Not stated."),
        ],
    },
]


def main() -> None:
    text_dir = ROOT / "text"
    golden_dir = ROOT / "golden"
    text_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    for doc in DOCS:
        doc_id = doc["doc_id"]
        pages = doc["pages"]
        (text_dir / f"{doc_id}.json").write_text(
            json.dumps({"doc_id": doc_id, "pages": pages}, indent=2) + "\n",
            encoding="utf-8",
        )
        (golden_dir / f"{doc_id}.json").write_text(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "source": doc["source"],
                    "pages": len(pages),
                    "scanned": doc["scanned"],
                    "fields": doc["fields"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"wrote {len(DOCS)} documents to {text_dir} and {golden_dir}")


if __name__ == "__main__":
    main()
