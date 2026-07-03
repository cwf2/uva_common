'''osf.py - downloading files from the OSF archive

Only reads from public storage, so no credentials are needed here. If
write access (upload) is ever added, osfclient's OSF() picks up
credentials from the OSF_USERNAME / OSF_PASSWORD (token) env vars.
'''

import os
from osfclient import OSF
from uva_common.config import CONFIG


def _resolve_node(node_id):
    '''Accept either a raw OSF node id or a key in CONFIG["osf"]["components"].'''
    components = CONFIG["osf"]["components"]
    return components.get(node_id, node_id)


def _storage(node_id):
    osf = OSF()
    project = osf.project(_resolve_node(node_id))
    return project.storage()


def download(filenames, node_id, local_dir=None):
    '''Download specific files by name from an OSF component.

    filenames — a filename, or list of filenames, to match by basename
    node_id   — an OSF node id, or a key in CONFIG["osf"]["components"]
    local_dir — destination directory (defaults to CONFIG["data_dir"])
    '''
    if isinstance(filenames, str):
        filenames = [filenames]
    local_dir = local_dir or CONFIG["data_dir"]

    for file in _storage(node_id).files:
        if file.name in filenames:
            dest = os.path.join(local_dir, file.name)
            with open(dest, "wb") as fh:
                file.write_to(fh)


def download_all(node_id, folder=None, local_dir=None):
    '''Download every file in an OSF component, preserving its folder structure.

    node_id   — an OSF node id, or a key in CONFIG["osf"]["components"]
    folder    — if given, only download files under this remote folder
    local_dir — destination directory (defaults to CONFIG["data_dir"]);
                remote subfolders are recreated under this directory
    '''
    local_dir = local_dir or CONFIG["data_dir"]
    prefix = f"{folder.strip('/')}/" if folder else None

    for file in _storage(node_id).files:
        remote_path = file.path.lstrip("/")
        if prefix and not remote_path.startswith(prefix):
            continue

        dest = os.path.join(local_dir, remote_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            file.write_to(fh)
