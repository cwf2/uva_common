'''hf_models.py - resolve the latest available wheel for a HuggingFace-hosted
spaCy model.

Models like grc_dep_web_trf/la_core_web_trf get frequent point-release
bumps, and the wheel URL is versioned in its filename, so a URL hardcoded
into CONFIG drifts out of date silently. This queries the HF Hub's public
API directly (no auth needed for a public repo) to find the current latest
wheel instead of requiring someone to notice a new release by hand.
'''
import re

import requests

WHEEL_RE = r"^{model_name}-(\d+(?:\.\d+)*)-py3-none-any\.whl$"


def latest_wheel_url(repo_id, model_name=None):
    '''Return the download URL for a model's latest available wheel on HF.

    repo_id    — HuggingFace repo id, e.g. "latincy/la_core_web_trf"
    model_name — wheel filename prefix; defaults to repo_id's last segment
    '''
    if model_name is None:
        model_name = repo_id.split("/")[-1]

    resp = requests.get(f"https://huggingface.co/api/models/{repo_id}")
    resp.raise_for_status()
    siblings = resp.json().get("siblings", [])

    pattern = re.compile(WHEEL_RE.format(model_name=re.escape(model_name)))
    versions = []
    for sib in siblings:
        m = pattern.match(sib["rfilename"])
        if m:
            version = tuple(int(p) for p in m.group(1).split("."))
            versions.append((version, sib["rfilename"]))

    if not versions:
        raise ValueError(f"No wheel found for '{model_name}' in {repo_id}")

    _, filename = max(versions)
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
