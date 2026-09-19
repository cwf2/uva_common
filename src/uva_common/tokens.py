'''tokens.py - loading token tables

Token tables are archived as the parser's output essentially unedited
(including punctuation rows), so provenance stays simple: what's on OSF is
what the model produced, plus the documented lemma/accent corrections and
the pos-mistag fix for punctuation-only tokens. Most analysis wants
punctuation filtered out, though, so load_tokens() does that at read time
by default -- never by altering the archived data -- and it can be turned
off to get the table exactly as saved.
'''

import glob
import os
import unicodedata

import pandas as pd

# read_csv would otherwise upcast cluster/turn to float64 (int-with-NA) and
# numeric-infer person, which is a categorical code (1/2/3), not a quantity
_DTYPE_FIXES = {
    "cluster": pd.Int64Dtype(),
    "turn": pd.Int64Dtype(),
    "person": pd.StringDtype(),
}


def load_tokens(paths, filter_punct=True):
    '''Read one or more token-table CSVs into a single DataFrame.

    paths        — a single path, a list of paths, or a directory (all
                    *.csv files directly inside it are read)
    filter_punct — if True (the default), drop PUNCT rows and reparent any
                    surviving token's head_id that pointed at a removed
                    one, so head_id always resolves to a row present in
                    the result. Pass False for the archived table exactly
                    as saved.
    '''
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.csv"))))
        else:
            files.append(p)

    df = pd.concat(
        (pd.read_csv(f, dtype=_DTYPE_FIXES) for f in files),
        ignore_index=True,
    )

    if filter_punct:
        df = _filter_punct(df)

    return df


def is_punct_text(text):
    '''True if text is non-empty and made up entirely of punctuation/symbol
    characters. Used instead of trusting the model's own pos=="PUNCT" tag,
    which turns out to be unreliable in both directions: some tokens whose
    text is pure punctuation get tagged with a real content POS, and (at
    least in la_core_web_trf) a handful of ordinary words -- e.g. morsu,
    currus, petit -- get mistagged PUNCT outright. Filtering on the
    surface text instead means load_tokens() never silently drops a real
    word, and works the same whether or not a given archived file has
    already had the export-time pos-relabeling fix applied.
    '''
    return isinstance(text, str) and text != "" and all(
        unicodedata.category(c).startswith(("P", "S")) for c in text
    )


def _filter_punct(df):
    '''Drop punctuation rows (by surface text, see is_punct_text), reparenting
    their dependents to the nearest surviving ancestor so every remaining
    head_id resolves to a row still present in the result. The ROOT row
    (head_id == tok_id) is never dropped, even if it's punctuation, so the
    walk always terminates.

    Tables from before tok_id/head_id existed are filtered with a plain
    drop, since there's nothing to reparent.
    '''
    if "tok_id" not in df.columns or "head_id" not in df.columns:
        return df.loc[~df["text"].apply(is_punct_text)].reset_index(drop=True)

    is_root = df["tok_id"] == df["head_id"]
    drop = df["text"].apply(is_punct_text) & ~is_root
    dropped = set(df.loc[drop, "tok_id"])
    head_of = dict(zip(df["tok_id"], df["head_id"]))

    def resolve(head_id):
        seen = set()
        while head_id in dropped and head_id not in seen:
            seen.add(head_id)
            head_id = head_of[head_id]
        return head_id

    kept = df.loc[~drop].copy()
    kept["head_id"] = kept["head_id"].apply(resolve)
    return kept.reset_index(drop=True)
