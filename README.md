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

See the ["Tokenized Texts" OSF component](https://osf.io/yahps/) to browse the files directly.

## Build Token Tables.ipynb

Builds the corpus from scratch: sources Greek epic texts (Homer, Apollonius, Quintus, Triphiodorus, Nonnus) from [Perseus](https://github.com/perseusDL/canonical-greekLit), parses them with [OdyCy](https://huggingface.co/chcaa/grc_odycy_joint_trf), and annotates speech/narration using the [DICES](https://dices.mta.ca/) database. Exports one token table per text to `data/tokens/`, which then get uploaded to the OSF archive above. Requires the `nlp` extra (see install instructions).