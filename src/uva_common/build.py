'''build.py - build the token table for a single text

build_tokens() is the one-text version of what "Build Token Tables.ipynb"
does for the whole canon: parse a Perseus edition with the language's spaCy
model, tidy up the parser's output, optionally label lines that fall inside
DICES speeches, and optionally write the CSV.

It needs a local Perseus mirror (see uva_common.clone_repo) and, for
speech annotation, the dices-client package and network access to DICES.
For a text that isn't in DICES, pass annotate_speech=False.
'''

import logging
import os
import re
import warnings

import pandas as pd

from uva_common.config import CONFIG
from uva_common.text import Text
from uva_common.tokens import is_punct_text

log = logging.getLogger(__name__)

# Odysseus' apologue is distinguished from his other speeches
_APOLOGUE_IDS = ["3F63", "A2AB"]

_CORRECTION_FILES = ["accent_corrections.tsv", "lemma_corrections.tsv"]

# where DICES line numbers and a Perseus edition disagree (see load_speech_fixes)
_SPEECH_FIXES_FILE = "speech_line_fixes.tsv"

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


def load_speech_fixes():
    '''Read CONFIG["data_dir"]/speech_line_fixes.tsv: where a Perseus edition's
    line numbering is offset from DICES's, as a list of dicts with keys
    work_urn, book, from_line, to_line, offset, speech_id ("*" for any speech)
    and reason. An endpoint of a matching speech whose line number lies in
    [from_line, to_line] is moved by offset. Blank lines and "#" comments are
    skipped; a missing file means no fixes.'''
    path = os.path.join(CONFIG["data_dir"], _SPEECH_FIXES_FILE)
    if not os.path.exists(path):
        return []
    fixes = []
    with open(path, encoding="utf-8") as f:
        header = None
        for row in f:
            if not row.strip() or row.startswith("#"):
                continue
            fields = row.rstrip("\n").split("\t")
            if header is None:
                header = fields
                continue
            rec = dict(zip(header, fields))
            for k in ("from_line", "to_line", "offset"):
                rec[k] = int(rec[k])
            fixes.append(rec)
    return fixes


_LOCUS = re.compile(r"^(?:(\d+)\.)?(\d+)([A-Za-z]*)$")


def _locus_key(locus):
    '''Sortable (book, line, suffix) for a locus like "6.184a" or "212";
    None for loci that don't fit (e.g. "74_75").'''
    m = _LOCUS.match(locus)
    if m is None:
        return None
    return (int(m.group(1) or 0), int(m.group(2)), m.group(3))


def _apply_fixes(fixes, urn, speech_id, locus):
    '''Move one speech endpoint (e.g. "4.850") by any matching offset fix.'''
    key = _locus_key(locus)
    if key is None or key[2]:
        return locus
    for fx in fixes:
        if fx["work_urn"] != urn or fx["speech_id"] not in ("*", speech_id):
            continue
        if key[0] == int(fx["book"] or 0) and fx["from_line"] <= key[1] <= fx["to_line"]:
            book = f"{key[0]}." if key[0] else ""
            return f"{book}{key[1] + fx['offset']}"
    return locus


def annotate_speeches(tokens, urn, speeches):
    '''Add the SPEECH_COLUMNS to a token table, one value per line shared by
    all of that line's tokens. Lines outside any speech get level 0 and
    empty strings elsewhere.

    DICES endpoints that name a line the edition doesn't have are clamped
    inward -- a first line moves forward to the next line present, a last
    line back to the previous one -- after applying any offsets from
    speech_line_fixes.tsv. A speech left with no lines is skipped with a
    warning.'''
    line_table = pd.DataFrame({
        "urn": tokens["urn"].drop_duplicates()
    }).reset_index(drop=True)
    line_table = line_table.assign(
        speech_id=None, speaker=None, addressee=None,
        level=0, type=None, cluster=None, turn=None, tags=None,
    )
    loci = [u.rsplit(":", 1)[-1] for u in line_table["urn"]]
    keys = [_locus_key(l) for l in loci]
    position = {l: i for i, l in enumerate(loci)}
    fixes = load_speech_fixes()

    def find(locus, forward):
        if locus in position:
            return position[locus]
        target = _locus_key(locus)
        if target is None:
            return None
        same_book = [i for i, k in enumerate(keys) if k is not None and k[0] == target[0]]
        if forward:
            hits = [i for i in same_book if keys[i] >= target]
            found = min(hits) if hits else None
        else:
            hits = [i for i in same_book if keys[i] <= target]
            found = max(hits) if hits else None
        if found is not None:
            log.info("%s: %s not in edition; using %s", urn, locus, loci[found])
        return found

    for speech in speeches:
        if speech.work.urn != urn:
            continue

        sid = speech._attributes["public_id"]
        l_fi = _apply_fixes(fixes, urn, sid, speech.l_fi)
        l_la = _apply_fixes(fixes, urn, sid, speech.l_la)

        i_first, i_last = find(l_fi, True), find(l_la, False)
        if i_first is None or i_last is None or i_first > i_last:
            warnings.warn(
                f"speech {sid} ({speech.l_fi}-{speech.l_la}) has no lines in {urn}"
            )
            continue

        line_table.loc[i_first:i_last, "speech_id"] = sid
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
