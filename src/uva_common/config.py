CONFIG = {
    # location of data files in local filesystem
    "data_dir": "data",

    # XML namespaces used by Perseus TEI, needed for xpath
    "nsmap": {
        "cts": "http://chs.harvard.edu/xmlns/cts",
        "tei": "http://www.tei-c.org/ns/1.0",
        "py": "http://codespeak.net/lxml/objectify/pytype",
    },

    # Perseus corpus repositories
    "repositories": {
        "greek": {
            "label": "Perseus Greek",
            "base_url": "https://github.com/perseusDL",
            "repo_name": "canonical-greekLit",
            "commit": "beed7ea8926266ad90935c183f2bcf2caf4ac0dc",
        },
        "latin": {
            "label": "Perseus Latin",
            "base_url": "https://github.com/perseusDL",
            "repo_name": "canonical-latinLit",
            "commit": "b90226f0b3e5fe1ac1dc38bd788e7854319149e1",
        },
    },

    # Fixed canon of six hexameter epics for the CCC2026/Lotte pipeline.
    # (name, urn) pairs — the English name is not reliably derivable from
    # Perseus CTS metadata (missing, mislabeled, or non-"eng" language tags
    # across these editions), so it's given here by hand.
    "texts": [
        ("Iliad", "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2"),
        ("Odyssey", "urn:cts:greekLit:tlg0012.tlg002.perseus-grc2"),
        ("Argonautica", "urn:cts:greekLit:tlg0001.tlg001.perseus-grc2"),
        ("Posthomerica", "urn:cts:greekLit:tlg2046.tlg001.perseus-grc2"),
        ("Sack of Troy", "urn:cts:greekLit:tlg0647.tlg001.perseus-grc2"),
        ("Dionysiaca", "urn:cts:greekLit:tlg2045.tlg001.perseus-grc2"),
    ],

    # OSF archive: "UvA Digital Classics" project and its components
    "osf": {
        "project": "ntdgv",
        "components": {
            "tokens": "yahps",  # "Tokenized Texts"
        },
    },

    # spaCy models for Greek and Latin
    "spacy_models": {
        "greek": {
            "model_name": "grc_odycy_joint_trf",
            "location": "https://huggingface.co/chcaa/grc_odycy_joint_trf/resolve/main/grc_odycy_joint_trf-0.7.0-py3-none-any.whl",
        },
        "latin": {
            "model_name": "la_core_web_trf",
            "location": "https://huggingface.co/latincy/la_core_web_trf/resolve/main/la_core_web_trf-3.9.5-py3-none-any.whl",
        },
    },
}
