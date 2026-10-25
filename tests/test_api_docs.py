"""docs/api.md must only name objects that exist.

``mkdocs build --strict`` does not catch this: mkdocstrings silently skips a
``members:`` entry that does not resolve.
"""

import importlib
import re
from pathlib import Path

API_DOC = Path(__file__).resolve().parent.parent / 'docs' / 'api.md'


def documented_names():
    """Yield ``(target, member)`` for every ``:::`` block; *member* is None for the target itself."""
    target = None
    for line in API_DOC.read_text(encoding='utf-8').splitlines():
        if line.startswith('::: '):
            target = line[4:].strip()
            yield target, None
        elif line and not line[0].isspace():
            target = None
        elif target and (match := re.fullmatch(r'\s+- (\w+)', line)):
            yield target, match.group(1)


def resolve(path):
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError:
        module, _, attr = path.rpartition('.')
        return getattr(importlib.import_module(module), attr)


def test_api_docs_name_only_existing_objects():
    missing = []
    for target, member in documented_names():
        try:
            obj = resolve(target)
        except (ImportError, AttributeError):
            missing.append(target)
            continue
        if member is not None and not hasattr(obj, member):
            missing.append(f'{target}.{member}')

    assert not missing, f'docs/api.md names objects that do not exist: {missing}'
