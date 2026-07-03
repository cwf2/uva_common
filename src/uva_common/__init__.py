'''uva_common — Tools for Digital Classics Research at the University of Amsterdam'''

import os
import git

from uva_common.config import CONFIG
from uva_common.text import Text, parse_urn, get_xml
from uva_common.osf import download, download_all


def clone_repo(language):
    '''Clone a Perseus text repository and reset to the pinned commit.'''
    rec = CONFIG["repositories"][language]
    dest_path = os.path.join(CONFIG["data_dir"], rec["repo_name"])
    remote_url = f"{rec['base_url']}/{rec['repo_name']}.git"

    if os.path.exists(dest_path):
        repo = git.Repo(dest_path)
    else:
        print(f" - retrieving {dest_path}")
        repo = git.Repo.clone_from(remote_url, dest_path)

    print(f" - resetting to commit {rec['commit']}")
    repo.head.reset(rec["commit"], working_tree=True)
