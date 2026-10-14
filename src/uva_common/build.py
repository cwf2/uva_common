'''build.py - build the token table for a single text

build_tokens() is the one-text version of what "Build Token Tables.ipynb"
does for the whole canon: parse a Perseus edition with the language's spaCy
model, tidy up the parser's output, optionally label lines that fall inside
DICES speeches, and optionally write the CSV.

It needs a local Perseus mirror (see uva_common.clone_repo) and, for
speech annotation, the dices-client package and network access to DICES.
For a text that isn't in DICES, pass annotate_speech=False.
'''

import os
import warnings

import pandas as pd

from uva_common.config import CONFIG
from uva_common.text import Text
from uva_common.tokens import is_punct_text

# Known mismatches between DICES line numbers and the Perseus edition:
# (work URN, which field, DICES value) -> the value that exists in the edition
_SPEECH_LINE_FIXES = {
    ("urn:cts:greekLit:tlg0001.tlg001.perseus-grc2", "l_la", "3.739"): "3.738",
    ("urn:cts:greekLit:tlg0012.tlg002.perseus-grc2", "l_fi", "10.456"): "10.455",
}

# Odysseus' apologue is distinguished from his other speeches
_APOLOGUE_IDS = ["3F63", "A2AB"]

_CORRECTION_FILES = ["accent_corrections.tsv", "lemma_corrections.tsv"]

# columns that annotate_speeches() adds
SPEECH_COLUMNS = ["speech_id", "speaker", "addressee", "level", "type",
                  "cluster", "turn", "tags"]


def build_tokens(urn, name=None, out_dir=None, annotate_speech=True,
                 corrections=None, speeches=None, nlp_pipeline=None):
    '''Parse one text and return its token table as a DataFrame.

    urn             — CTS URN of the edition, e.g.
                       "urn:cts:greekLit:tlg4081.tlg001.perseus-grc2"
    name            — English work name for the "work" column; defaults to
                       the title in the Perseus metadata, which may not be
                       English
    out_dir         — if given, also write the table there as
                       {workgroup}.{work}.{edition}.csv (the archive's
                       filename convention); the directory is created
    annotate_speech — label lines inside DICES speeches (adds the
                       speech_id/speaker/addressee/level/type/cluster/turn/
                       tags columns). Needs the dices-client package and
                       the text must be in DICES.
    corrections     — apply the lemma and accent corrections files from
                       CONFIG["data_dir"]. None (the default) means "yes for
                       Greek, no for Latin", since the files are Greek-only.
    speeches        — pre-fetched DICES speech objects, to avoid a fetch per
                       text when building many; only those belonging to this
                       text are used
    nlp_pipeline    — passed through to Text.parse() (default: the model
                       CONFIG names for the text's language)

    Punctuation rows are kept (see tokens.load_tokens for filtering at read
    time); punctuation-only tokens the model mistagged with a content POS
    are re-tagged PUNCT.
    '''
    text = Text(urn)

    # fetch speeches first, so a text missing from DICES fails before the
    # (slow) parse rather than after it
    if annotate_speech and speeches is None:
        speeches = fetch_speeches(urn)

    tokens = text.parse(nlp_pipeline)

    mask = tokens["text"].apply(is_punct_text)
    tokens.loc[mask, "pos"] = "PUNCT"

    if corrections is None:
        corrections = text.language == "greek"
    if corrections:
        tokens = apply_corrections(tokens)

    if annotate_speech:
        tokens = annotate_speeches(tokens, urn, speeches)

    tokens.insert(2, "work", name if name is not None else text.title)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{text.workgroup}.{text.work}.{text.edition}.csv")
        tokens.to_csv(path, index=False)

    return tokens


def apply_corrections(tokens):
    '''Apply the lemma-correction tables (old<TAB>new, "#" comments) from
    CONFIG["data_dir"] to the lemma column; returns the table.'''
    for fn in _CORRECTION_FILES:
        path = os.path.join(CONFIG["data_dir"], fn)
        if not os.path.exists(path):
            warnings.warn(f"{path} not found; skipping that correction file")
            continue
        with open(path) as f:
            for row in f:
                if not row.startswith("#"):
                    old, new = row.strip().split("\t")
                    tokens.loc[tokens["lemma"] == old, "lemma"] = new
    return tokens


def fetch_speeches(urn):
    '''All DICES speeches belonging to the work with this URN.'''
    from dicesapi import DicesAPI

    api = DicesAPI()
    works = [w for w in api.getWorks() if w.urn == urn]
    if not works:
        raise ValueError(
            f"{urn} is not in DICES; pass annotate_speech=False to build "
            "its token table without speech annotation."
        )
    return [s for s in api.getSpeeches(work_id=works[0].id) if s.work.urn == urn]


def annotate_speeches(tokens, urn, speeches):
    '''Add the SPEECH_COLUMNS to a token table, one value per line shared by
    all of that line's tokens. Lines outside any speech get level 0 and
    empty strings elsewhere; speeches whose first or last line can't be
    found in the table are skipped with a warning.'''
    line_table = pd.DataFrame({
        "urn": tokens["urn"].drop_duplicates()
    }).reset_index(drop=True)
    line_table = line_table.assign(
        speech_id=None, speaker=None, addressee=None,
        level=0, type=None, cluster=None, turn=None, tags=None,
    )

    for speech in speeches:
        if speech.work.urn != urn:
            continue

        l_fi, l_la = speech.l_fi, speech.l_la
        l_fi = _SPEECH_LINE_FIXES.get((urn, "l_fi", l_fi), l_fi)
        l_la = _SPEECH_LINE_FIXES.get((urn, "l_la", l_la), l_la)

        matches_first = line_table.index[line_table["urn"] == f"{urn}:{l_fi}"]
        matches_last = line_table.index[line_table["urn"] == f"{urn}:{l_la}"]
        if len(matches_first) == 0 or len(matches_last) == 0:
            warnings.warn(
                f"could not locate speech {speech._attributes['public_id']} "
                f"({l_fi}-{l_la}) in {urn}"
            )
            continue

        i_first, i_last = matches_first[0], matches_last[0]
        line_table.loc[i_first:i_last, "speech_id"] = speech._attributes["public_id"]
        line_table.loc[i_first:i_last, "speaker"] = speech.getSpkrString()
        line_table.loc[i_first:i_last, "addressee"] = speech.getAddrString()
        line_table.loc[i_first:i_last, "level"] = speech.level + 1
        line_table.loc[i_first:i_last, "type"] = speech.type
        line_table.loc[i_first:i_last, "cluster"] = speech.cluster.id
        line_table.loc[i_first:i_last, "turn"] = speech.part
        line_table.loc[i_first:i_last, "tags"] = ";".join(
            tag["type"] for tag in speech._attributes["tags"]
        )

    line_table.loc[line_table["speech_id"].isin(_APOLOGUE_IDS), "speaker"] = "Odysseus-Apologue"

    return pd.merge(tokens, line_table, on="urn")
