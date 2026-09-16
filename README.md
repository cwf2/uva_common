# uva_common

Shared workflows for Digital Classics research at the University of Amsterdam.

To install from GitHub: `pip install git+https://github.com/cwf2/uva_common`

The NLP pipeline (spaCy) is an optional dependency, since most consumers only need the pre-built token tables, not to re-parse texts themselves. To install it: `pip install "uva_common[nlp] @ git+https://github.com/cwf2/uva_common"`

## Downloading pre-built token tables

Most projects don't need to re-run the parsing pipeline — a token table for each text in the corpus is already archived on OSF and can be pulled down directly:

```python
import uva_common
uva_common.download_all("tokens")  # -> data/tokens/*.csv
```

See the ["Tokenized Texts" OSF component](https://osf.io/yahps/) to browse the files directly, or open [`Load Tokens.ipynb`](https://colab.research.google.com/github/cwf2/uva_common/blob/main/Load%20Tokens.ipynb) in Colab for a minimal runnable example.

The corpus is six Greek hexameter epics: Homer's *Iliad* and *Odyssey*, Apollonius' *Argonautica*, Quintus' *Posthomerica*, Triphiodorus' *Sack of Troy*, and Nonnus' *Dionysiaca* — one CSV per text, named `{workgroup}.{work}.{edition}.csv` after its CTS URN (e.g. `tlg0012.tlg001.perseus-grc2.csv` for the Iliad).

### Token table format

One row per token (punctuation already removed). Location and surface form:

| Column | Meaning |
|---|---|
| `author`, `title`, `work` | Text's author and title (original language), and its English name |
| `urn` | CTS URN of the token's verse line |
| `line_id` | `{book}_{line}`, zero-padded (e.g. `01_0001`) |
| `text` | Surface form |
| `lemma` | Dictionary form, after manual accent/elision corrections |

Morphology, from spaCy's [`grc_dep_web_trf`](https://huggingface.co/latincy/grc_dep_web_trf) (Universal Dependencies tagset — empty where a feature doesn't apply to that POS):

| Column | Meaning |
|---|---|
| `pos` | Universal POS tag |
| `verbform`, `mood`, `tense`, `voice`, `person`, `number`, `case`, `gender` | UD morphological features |

Speech annotation, from the [DICES](https://dices.mta.ca/) database — empty/NA for narrative tokens (a token only has these if it falls inside a speech), **except `level`**, which (a) is always set, and (b) differs from DICES by +1: narrative has `level==0`, direct speech is `level==1`, and embedded speech is `level>=2`.

| Column | Meaning |
|---|---|
| `speech_id` | DICES speech's public ID |
| `speaker`, `addressee` | Speaker/addressee string. `speaker` is set to the sentinel `"Odysseus-Apologue"` for Odysseus' first-person narration in *Odyssey* 9–12, distinguishing it from his other speeches |
| `level` | Nesting depth, always set: `0` for narrative, `1` for direct speech, `2`+ for speech nested inside speech |
| `type` | DICES speech type |
| `cluster` | ID of the speech's dialogue-exchange cluster |
| `turn` | Speech's turn/part index within its cluster |
| `tags` | Semicolon-separated DICES tags |

**Dtype note**: `pandas.read_csv` upcasts integer columns with NA values (`cluster`, `turn`) to `float64`, and numeric-infers `person` (which is a categorical code — 1/2/3 — not a quantity) the same way. `Load Tokens.ipynb` shows the fix: read with `dtype={"cluster": pd.Int64Dtype(), "turn": pd.Int64Dtype(), "person": pd.StringDtype()}`.

## Build Token Tables.ipynb

Builds the corpus from scratch: sources Greek epic texts (Homer, Apollonius, Quintus, Triphiodorus, Nonnus) from [Perseus](https://github.com/perseusDL/canonical-greekLit), parses them with [grc_dep_web_trf](https://huggingface.co/latincy/grc_dep_web_trf), and annotates speech/narration using the [DICES](https://dices.mta.ca/) database. Exports one token table per text to `data/tokens/`, which then get uploaded to the OSF archive above. Requires the `nlp` extra (see install instructions).