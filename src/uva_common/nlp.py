import bisect
import spacy
import pandas as pd
from uva_common.config import CONFIG

# cache loaded models so each one is only loaded once per session
_model_cache = {}


def pipeline_picker(spec):
    '''Return a spaCy pipeline, loading and caching it if necessary.

    spec can be:
      - a Text or other object with a "language" attribute
      - a language string recognised in CONFIG["spacy_models"] (e.g. "greek")
      - a model name string passed directly to spacy.load()
    '''
    if hasattr(spec, "language"):
        model_name = CONFIG["spacy_models"][spec.language]["model_name"]
    elif spec in CONFIG["spacy_models"]:
        model_name = CONFIG["spacy_models"][spec]["model_name"]
    else:
        model_name = spec

    if model_name not in _model_cache:
        _model_cache[model_name] = spacy.load(model_name)

    return _model_cache[model_name]


def line_array_to_token_table(lines, nlp_pipeline):
    '''Parse a line array with a spaCy pipeline; return a token DataFrame.

    lines        — list of dicts as returned by text._parse_lines()
    nlp_pipeline — a loaded spaCy Language object (e.g. from pipeline_picker())

    Lines are joined into one string before parsing so the model has full
    cross-line context. Tokens are mapped back to their source line via the
    cumsum offsets stored in each line dict.
    '''
    one_long_string = " ".join(l["text"] for l in lines)
    doc = nlp_pipeline(one_long_string)

    line_offsets = [l["cumsum"] for l in lines]

    tokens = []
    for token in doc:
        i = bisect.bisect_right(line_offsets, token.idx) - 1
        tokens.append(dict(
            urn=lines[i]["urn"],
            line_id=lines[i]["line_id"],
            text=token.text,
            lemma=token.lemma_,
            pos=token.pos_,
            verbform=";".join(token.morph.get("VerbForm")),
            mood=";".join(token.morph.get("Mood")),
            tense=";".join(token.morph.get("Tense")),
            voice=";".join(token.morph.get("Voice")),
            person=";".join(token.morph.get("Person")),
            number=";".join(token.morph.get("Number")),
            case=";".join(token.morph.get("Case")),
            gender=";".join(token.morph.get("Gender")),
        ))

    return pd.DataFrame(tokens)
