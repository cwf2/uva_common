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


def load_tokens(paths, filter_punct=True, include_bracketed=False):
    '''Read one or more token-table CSVs into a single DataFrame.

    paths             — a single path, a list of paths, or a directory (all
                         *.csv files directly inside it are read)
    filter_punct      — if True (the default), drop PUNCT rows
    include_bracketed — if False (the default), also drop tokens the edition
                         brackets or deletes (the "bracketed" column; tables
                         built before that column existed have none)

    Whatever is dropped, any surviving token's head_id that pointed at a
    removed one is reparented to its nearest surviving ancestor, so head_id
    always resolves to a row present in the result (a token whose whole
    ancestry was removed becomes a root, with head_id == tok_id). Pass
    filter_punct=False, include_bracketed=True for the archived table
    exactly as saved.
    '''
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.csv"))))
        else:
            files.append(p)

    # Each file is read and filtered on its own: tok_id is only unique within
    # a text (every text has a 01_00000), so dropping and reparenting across
    # concatenated texts would mistake one text's tokens for another's.
    frames = []
    for f in files:
        # only the empty string means "missing": pandas' default NA strings
        # would silently turn the real Latin lemma "null" (nulli, of nullus)
        # into NaN
        df = pd.read_csv(f, dtype=_DTYPE_FIXES, keep_default_na=False, na_values=[""])
        if "bracketed" in df.columns:
            df["bracketed"] = df["bracketed"].fillna(False).astype(bool)
        frames.append(_drop_rows(df, punct=filter_punct, bracketed=not include_bracketed))

    df = pd.concat(frames, ignore_index=True)
    if "bracketed" in df.columns:
        # tables built before the column existed contribute NaN
        df["bracketed"] = df["bracketed"].fillna(False).astype(bool)
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


def _drop_rows(df, punct, bracketed):
    '''Drop punctuation rows (by surface text, see is_punct_text) and/or
    bracketed rows, reparenting dependents of removed tokens to the nearest
    surviving ancestor (see load_tokens).

    A punctuation ROOT row (head_id == tok_id) is never dropped, so the
    walk up the tree always terminates there; a bracketed root can be, and
    its surviving dependents are promoted to roots.

    Tables from before tok_id/head_id existed are filtered with a plain
    drop, since there's nothing to reparent.
    '''
    drop = pd.Series(False, index=df.index)
    has_tree = "tok_id" in df.columns and "head_id" in df.columns
    if punct:
        is_punct = df["text"].apply(is_punct_text)
        if has_tree:
            is_punct &= df["tok_id"] != df["head_id"]
        drop |= is_punct
    if bracketed and "bracketed" in df.columns:
        drop |= df["bracketed"].astype(bool)

    if not has_tree:
        return df.loc[~drop].reset_index(drop=True)
    if not drop.any():
        return df.reset_index(drop=True)

    dropped = set(df.loc[drop, "tok_id"])
    head_of = dict(zip(df["tok_id"], df["head_id"]))

    def resolve(tok_id, head_id):
        seen = set()
        while head_id in dropped:
            if head_id in seen:
                return tok_id
            seen.add(head_id)
            parent = head_of[head_id]
            if parent == head_id:  # reached a dropped root
                return tok_id
            head_id = parent
        return head_id

    kept = df.loc[~drop].copy()
    kept["head_id"] = [resolve(t, h) for t, h in zip(kept["tok_id"], kept["head_id"])]
    return kept.reset_index(drop=True)
