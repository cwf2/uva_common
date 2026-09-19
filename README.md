# uva_common

Shared workflows for Digital Classics research at the University of Amsterdam.

To install from GitHub: `pip install git+https://github.com/cwf2/uva_common`

The NLP pipeline (spaCy) is an optional dependency, since most consumers only need the pre-built token tables, not to re-parse texts themselves. To install it: `pip install "uva_common[nlp] @ git+https://github.com/cwf2/uva_common"`

## Downloading pre-built token tables

Most projects don't need to re-run the parsing pipeline — a token table for each text in the corpus is already archived on OSF and can be pulled down directly:

```python
import uva_common
uva_common.download_all("tokens")  # -> data/tokens/*.csv
tokens = uva_common.load_tokens("data/tokens")  # one DataFrame, punctuation filtered out
```

See the ["Tokenized Texts" OSF component](https://osf.io/yahps/) to browse the files directly, or open [`Load Tokens.ipynb`](https://colab.research.google.com/github/cwf2/uva_common/blob/main/Load%20Tokens.ipynb) in Colab for a minimal runnable example.

### `load_tokens(paths, filter_punct=True)`

Reads one or more token-table CSVs (`paths` can be a single file, a list of files, or a directory) into one DataFrame, with the correct dtypes for `cluster`/`turn`/`person` (see the dtype note below — `load_tokens` applies this fix automatically, so it's only relevant if you read the CSVs some other way).

The archived CSVs are close to the parser's raw output — punctuation tokens are **not** removed before archiving, so provenance stays simple: what's on OSF is what the model produced, plus the documented lemma/accent corrections. `load_tokens` filters punctuation out at read time instead, since that's what nearly every downstream use wants:

- Punctuation rows are identified by surface text (is the token made up entirely of punctuation/symbol characters?), not by trusting the model's own `pos == "PUNCT"` tag — that tag turns out to be unreliable in both directions across the two models (a handful of ordinary Latin words like *petit*, *morsu*, *currus* get mistagged `PUNCT` by `la_core_web_trf`; some bare commas/apostrophes get mistagged with a real content POS by `grc_dep_web_trf`). Filtering on the text itself means a real word is never silently dropped.
- Any surviving token whose `head_id` pointed at a removed punctuation row gets reparented to that row's own head (walking up past a run of several punctuation tokens if need be), so `head_id` always resolves to a row present in the result. The `ROOT` row is never dropped, even if it's mistagged `PUNCT`, which guarantees the walk terminates.

Pass `filter_punct=False` to get the table exactly as archived, punctuation and all.

The corpus is six Greek hexameter epics: Homer's *Iliad* and *Odyssey*, Apollonius' *Argonautica*, Quintus' *Posthomerica*, Triphiodorus' *Sack of Troy*, and Nonnus' *Dionysiaca* — one CSV per text, named `{workgroup}.{work}.{edition}.csv` after its CTS URN (e.g. `tlg0012.tlg001.perseus-grc2.csv` for the Iliad).

### Token table format

One row per token, as archived — **including punctuation** (see `load_tokens` above for filtering it out at read time). Location and surface form:

| Column | Meaning |
|---|---|
| `author`, `title`, `work` | Text's author and title (original language), and its English name |
| `urn` | CTS URN of the token's verse line |
| `line_id` | `{book}_{line}`, zero-padded (e.g. `01_0001`) |
| `text` | Surface form |
| `lemma` | Dictionary form, after manual accent/elision corrections |

Provenance — same value on every row of a given file, same pattern as `author`/`title`/`work`:

| Column | Meaning |
|---|---|
| `model` | The spaCy model that produced this table, as `{package}=={version}` (e.g. `grc_dep_web_trf==3.8.4`) — read from the loaded model's own metadata, not just requested from `CONFIG`, so it reflects what actually ran |
| `uva_common_commit` | Commit hash of the `uva_common` checkout that produced this table (from `uva_common.repo_commit()`), with a `+dirty` suffix if the working tree had uncommitted changes at parse time. `None`/empty if it couldn't be determined (e.g. a non-editable `pip install` from a git URL discards `.git` when it builds the wheel) |

Morphology and dependency parse, from spaCy's [`grc_dep_web_trf`](https://huggingface.co/latincy/grc_dep_web_trf)/[`la_core_web_trf`](https://huggingface.co/latincy/la_core_web_trf) (Universal Dependencies tagset — empty where a feature doesn't apply to that POS):

| Column | Meaning |
|---|---|
| `pos` | Universal POS tag |
| `verbform`, `mood`, `tense`, `voice`, `person`, `number`, `case`, `gender` | UD morphological features |
| `tok_id` | `{book}_{token.i, zero-padded}` — unique per book (`token.i` alone is only unique per book, i.e. per spaCy `Doc`) |
| `head_id` | `tok_id` of this token's dependency-parse head; equal to its own `tok_id` for the sentence `ROOT` |
| `dep` | Dependency relation to `head_id`. Note: `grc_dep_web_trf` labels most attributive adjectives `nmod` rather than `amod` — filtering on `pos == "ADJ"` and following `head_id` will find far more adjective-noun pairs than filtering on `dep == "amod"` alone |

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

**Dtype note**: `pandas.read_csv` upcasts integer columns with NA values (`cluster`, `turn`) to `float64`, and numeric-infers `person` (which is a categorical code — 1/2/3 — not a quantity) the same way. `load_tokens` applies the fix (`dtype={"cluster": pd.Int64Dtype(), "turn": pd.Int64Dtype(), "person": pd.StringDtype()}`) automatically; only worth doing by hand if you're reading a CSV some other way.

## Build Token Tables.ipynb

Builds the corpus from scratch: sources Greek epic texts (Homer, Apollonius, Quintus, Triphiodorus, Nonnus) from [Perseus](https://github.com/perseusDL/canonical-greekLit), parses them with [grc_dep_web_trf](https://huggingface.co/latincy/grc_dep_web_trf), and annotates speech/narration using the [DICES](https://dices.mta.ca/) database. Exports one token table per text to `data/tokens/`, which then get uploaded to the OSF archive above. Requires the `nlp` extra (see install instructions).