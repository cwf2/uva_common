import os
import re
import pandas as pd
from lxml import etree

from uva_common.config import CONFIG


def parse_urn(urn):
    '''Extract components from a CTS URN; returns a dict.'''
    m = re.match(
        r"urn:cts:(greek|latin)Lit:([a-z]+[0-9]+)\.([a-z]+[0-9]+)\.(.+)", urn
    )
    if m is None:
        raise ValueError(f"Invalid URN: {urn}")
    return dict(
        urn=urn,
        language=m.group(1),
        workgroup=m.group(2),
        work=m.group(3),
        edition=m.group(4),
    )


def repo_commit():
    '''uva_common's own current git commit hash, or None if it can't be
    determined (e.g. installed from a wheel with no .git directory --
    a plain `pip install` from a git+URL, as opposed to a real checkout
    or an editable install, discards .git when the wheel is built).

    A "+dirty" suffix means the working tree had uncommitted changes at
    the moment this was called, so the commit hash alone doesn't fully
    pin the code that produced the result.
    '''
    try:
        import git
        repo = git.Repo(os.path.dirname(__file__), search_parent_directories=True)
        commit = repo.head.commit.hexsha
        if repo.is_dirty(untracked_files=True):
            commit += "+dirty"
        return commit
    except Exception:
        return None


def get_xml(urn):
    '''Load TEI XML for a CTS URN from the local Perseus mirror.'''
    rec = parse_urn(urn)
    path = os.path.join(
        CONFIG["data_dir"],
        f"canonical-{rec['language']}Lit",
        "data", rec["workgroup"], rec["work"],
        f"{rec['workgroup']}.{rec['work']}.{rec['edition']}.xml",
    )
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")
    return etree.parse(path)


# elements that, directly beside a <del>, mark it as having been replaced
_REPLACEMENT_TAGS = {"add", "corr", "reg", "choice"}


def _local(el):
    '''Local tag name of an element, or None for comments/processing instructions.'''
    return etree.QName(el).localname if isinstance(el.tag, str) else None


def _adjacent_replacement(del_):
    '''True if a <del> is immediately followed or preceded by replacement
    markup (<add>, <corr>, <reg>, <choice>) with no text in between.
    Merely sharing a parent isn't enough: in prose, unrelated <add>s and
    <del>s share a paragraph.'''
    nxt, prv = del_.getnext(), del_.getprevious()
    if nxt is not None and _local(nxt) in _REPLACEMENT_TAGS and not (del_.tail or "").strip():
        return True
    if prv is not None and _local(prv) in _REPLACEMENT_TAGS and not (prv.tail or "").strip():
        return True
    return False


def _clean_xml(xml):
    '''Tidy a TEI element in place: remove editorial notes (apparatus), and
    the content of any <del> that has been replaced by adjacent markup.

    Other <del> content -- text the editor brackets or deletes without a
    stated replacement -- is deliberately kept; _parse_lines records where
    it is so the tokens can be flagged as bracketed.
    '''
    nsmap = CONFIG["nsmap"]
    for note in xml.findall(".//tei:note", namespaces=nsmap):
        note.clear(keep_tail=True)
    for del_ in xml.findall(".//tei:del", namespaces=nsmap):
        if _adjacent_replacement(del_):
            del_.clear(keep_tail=True)


def _get_books(xml, edition_urn):
    '''Return a list of book dicts for each subdivision of the edition,
    or the whole edition as a single entry if there are no subdivisions.'''
    nsmap = CONFIG["nsmap"]
    editions = xml.xpath(
        f'.//tei:div[@type="edition" and @n="{edition_urn}"]',
        namespaces=nsmap,
    )
    if len(editions) == 0:
        raise ValueError(f"No edition found for {edition_urn}")
    if len(editions) > 1:
        raise ValueError(f"Multiple editions matched {edition_urn}")
    edition = editions[0]

    books = edition.xpath(
        './/tei:div[@subtype="book" or @subtype="Book"]',
        namespaces=nsmap,
    )
    if len(books) == 0:
        return [{"prefix": "", "xml": edition}]
    return [{"prefix": b.get("n"), "xml": b} for b in books]


def _make_line_id(prefix, line_num):
    '''Build a sortable "BB_LLLL" id from a book prefix and a raw line number.

    Zero-pads the book prefix to 2 digits ("00" for texts with no book
    subdivisions) and the leading digit run of the line number to 4 digits,
    keeping a trailing "a"/"b" suffix (e.g. "568a") if present. Anything else
    after the leading digit run (e.g. editorial compounds like "74_75", used
    by Perseus to flag transposed lines) is dropped from the id — the
    original value is preserved in the "urn" locus, whose rightmost
    "."-separated segment is always the raw line number.
    '''
    pref_padded = (prefix or "00").zfill(2)
    m = re.match(r"(\d+)([ab])?", line_num)
    if m is None:
        return f"{pref_padded}_{line_num}"
    digits, letter = m.groups()
    return f"{pref_padded}_{digits.zfill(4)}{letter or ''}"


def _text_with_spans(el):
    '''Whitespace-normalized text of an element, plus the character spans of
    it that fall inside <del> elements.

    Returns (text, spans) where spans is a list of (start, end) offsets into
    text. Text outside the element (its own tail) is not included.
    '''
    chars = []  # (character, inside_del)

    def walk(e, in_del):
        if e.text:
            chars.extend((c, in_del) for c in e.text)
        for child in e:
            if isinstance(child.tag, str):
                walk(child, in_del or _local(child) == "del")
            if child.tail:
                chars.extend((c, in_del) for c in child.tail)

    walk(el, False)

    out, flags, prev_space = [], [], True
    for c, flagged in chars:
        if c.isspace():
            if prev_space:
                continue
            out.append(" ")
            flags.append(flagged)
            prev_space = True
        else:
            out.append(c)
            flags.append(flagged)
            prev_space = False
    if out and out[-1] == " ":
        out.pop()
        flags.pop()

    spans, start = [], None
    for k, flagged in enumerate(flags):
        if flagged and start is None:
            start = k
        elif not flagged and start is not None:
            spans.append((start, k))
            start = None
    if start is not None:
        spans.append((start, len(flags)))

    return "".join(out), spans


def _parse_lines(edition_urn, prefix, xml, cumsum_start=0):
    '''Extract verse lines from a TEI element into a list of dicts.

    cumsum_start allows the caller to continue a running character offset
    across multiple book elements. The cumsum values are consumed by
    line_array_to_token_table() to map spaCy token positions back to lines.

    Lines with no text (e.g. a <gap/> marking a lacuna) are skipped: an
    empty line would put two spaces in a row into the joined string, and
    some spaCy models (la_core_web_trf) collapse runs of whitespace, which
    would throw every later token in the book off its line.

    Each line dict also carries "deleted_spans": the (start, end) character
    offsets within its text that sit inside <del> elements.
    '''
    nsmap = CONFIG["nsmap"]
    lines = []
    cumsum = cumsum_start

    for l in xml.findall(".//tei:l", namespaces=nsmap):
        line_num = l.get("n")
        if line_num is None:
            continue

        line_text, deleted_spans = _text_with_spans(l)
        if not line_text:
            continue

        locus = f"{prefix}.{line_num}" if prefix else line_num
        lines.append(dict(
            urn=f"{edition_urn}:{locus}",
            line_id=_make_line_id(prefix, line_num),
            seq=len(lines),
            text=line_text,
            cumsum=cumsum,
            deleted_spans=deleted_spans,
        ))
        # +1 accounts for the space in " ".join() used before parsing
        cumsum += len(line_text) + 1

    return lines


class Text:
    '''A single text in the corpus, identified by CTS URN.

    Loads metadata and XML on construction. Call parse() to run the NLP
    pipeline and produce a token DataFrame. Results can be stored on the
    object as text.tokens by the caller.
    '''

    def __init__(self, urn):
        self.urn = urn
        rec = parse_urn(urn)
        self.language = rec["language"]
        self.workgroup = rec["workgroup"]
        self.work = rec["work"]
        self.edition = rec["edition"]

        self._xml = get_xml(urn)
        _clean_xml(self._xml)

        nsmap = CONFIG["nsmap"]
        title_el = self._xml.find(".//tei:titleStmt/tei:title", namespaces=nsmap)
        author_el = self._xml.find(".//tei:titleStmt/tei:author", namespaces=nsmap)
        self.title = title_el.text if title_el is not None else None
        self.author = author_el.text if author_el is not None else None

    def parse(self, nlp_pipeline=None):
        '''Run the NLP pipeline and return a token DataFrame.

        Iterates over books one at a time to keep peak memory bounded —
        each book's line array and spaCy Doc are discarded before the next
        book is processed. The returned DataFrame is the only output kept.

        nlp_pipeline — a loaded spaCy Language object, or anything accepted
                       by nlp.pipeline_picker() (language string, model name,
                       or object with a .language attribute). Defaults to the
                       model specified in CONFIG for this text's language.
        '''
        from uva_common.nlp import pipeline_picker, line_array_to_token_table

        if nlp_pipeline is None:
            nlp_pipeline = pipeline_picker(self)

        all_tokens = []
        for book in _get_books(self._xml, self.urn):
            lines = _parse_lines(self.urn, book["prefix"], book["xml"])
            if not lines:
                continue
            tokens = line_array_to_token_table(lines, nlp_pipeline)
            all_tokens.append(tokens)

        result = pd.concat(all_tokens, ignore_index=True)
        result.insert(0, "author", self.author)
        result.insert(1, "title", self.title)
        # provenance: the model's own self-reported identity (not just
        # CONFIG's request for it -- this is what actually ran), and the
        # uva_common commit whose code produced this table. Same value
        # on every row, same pattern as author/title.
        model_name = f"{nlp_pipeline.meta.get('lang', '?')}_{nlp_pipeline.meta.get('name', '?')}"
        result.insert(2, "model", f"{model_name}=={nlp_pipeline.meta.get('version', '?')}")
        result.insert(3, "uva_common_commit", repo_commit())
        return result

    def __repr__(self):
        return f"Text({self.author!r}, {self.title!r})"
