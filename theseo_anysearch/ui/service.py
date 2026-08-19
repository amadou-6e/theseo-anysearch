"""JSON command surface for the native UI backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workspace import scan_workspace, validate_configuration


def main() -> None:
    """Execute one explicit UI backend operation and print one JSON document."""

    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("scan", "validate"))
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    if arguments.operation == "scan":
        output = scan_workspace(arguments.path).model_dump(mode="json")
    else:
        output = {
            "valid": not (diagnostics := validate_configuration(arguments.path)),
            "diagnostics": [item.model_dump() for item in diagnostics],
        }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
