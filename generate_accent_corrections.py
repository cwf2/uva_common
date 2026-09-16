#!/usr/bin/env python3
'''Regenerate data/accent_corrections.tsv from the "cleaning up lexical
features.xlsx" spreadsheet's "cleaning up accent variations" sheet.

The spreadsheet itself isn't version-controlled (it's a personal file, not
part of any repo) — this script exists so the derivation from spreadsheet
to TSV is documented and reproducible if the spreadsheet is ever revised,
rather than living only as scratch notebook cells.

Each row maps up to four "old" forms (found by name, not column position —
see below) to one "preferred lemma". A form appearing in more than one row
resolves to whichever row is processed last (matches the original,
unintentional behavior of the notebook this was ported from; rows in the
spreadsheet did not have conflicting corrections as of the last check, so
this hasn't mattered in practice).

Ported from ccc2026/Untitled.ipynb, which had a bug: it selected the old-
form columns by fixed position (a hardcoded column-index slice), which
silently skipped the sheet's 4th old-form column ("Original lemma 3") —
36 of 512 rows were affected, dropping real corrections (e.g. Δηριάδης,
Τηλέμαχος, Σάτυρος never got merged with their preferred lemma). Naming
the columns explicitly here, rather than slicing by position, fixes that
and is robust to the sheet's columns ever being reordered.
'''
import argparse
import os

import pandas as pd

SHEET_NAME = "cleaning up accent variations"
OLD_FORM_COLUMNS = [
    "Lemma (no accents)",
    "Original Lemma 1",
    "Original Lemma 2",
    "Original lemma 3",
]
PREFERRED_COLUMN = "preferred lemma"

# Manual overrides: old forms to drop even though the spreadsheet maps them
# to a preferred lemma — for cases where the "correction" would actually
# merge two distinct words rather than reconcile spelling variants of one.
#
# ὠμός ("raw/cruel", adjective) vs ὦμος ("shoulder", noun): the spreadsheet
# maps both to ὦμος, but spaCy already lemmatizes them separately, so this
# row isn't fixing an accent inconsistency, it's collapsing two real
# lemmas. Excluding ὠμός here leaves spaCy's own lemmatization standing;
# ωμος (the unaccented form, a genuine spelling gap) still gets corrected.
# Pending: rerun tokenization and inspect what actually lemmatizes as
# ὠμός — if spaCy turns out to be misassigning real ὦμος (shoulder) tokens
# to ὠμός, revisit and merge both after all.
EXCLUDED_OLD_FORMS = {"ὠμός"}


def generate_corrections(xlsx_path):
    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)

    corrections = {}
    for col in OLD_FORM_COLUMNS:
        pairs = df[[col, PREFERRED_COLUMN]].dropna(subset=[col])
        for old, new in zip(pairs[col], pairs[PREFERRED_COLUMN]):
            corrections[old] = new

    for old in EXCLUDED_OLD_FORMS:
        corrections.pop(old, None)

    return corrections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xlsx_path", help="path to 'cleaning up lexical features.xlsx'")
    parser.add_argument(
        "--output", default=os.path.join(os.path.dirname(__file__), "data", "accent_corrections.tsv"),
        help="output TSV path (default: data/accent_corrections.tsv)",
    )
    args = parser.parse_args()

    corrections = generate_corrections(args.xlsx_path)
    with open(args.output, "w") as f:
        for old, new in corrections.items():
            f.write(f"{old}\t{new}\n")

    print(f"Wrote {len(corrections)} corrections to {args.output}")
