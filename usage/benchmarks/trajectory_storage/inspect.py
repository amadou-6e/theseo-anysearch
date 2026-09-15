"""Readable inspection/export for benchmark artifacts, not production replay files."""
import argparse
from pathlib import Path
from .codecs import FORMATS, json_bytes, read


def inspect(directory, kind, step=None):
    loaded = read(directory, kind)
    return loaded.export() if step is None else loaded.step(step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = json_bytes(inspect(args.directory, args.format, args.step), pretty=True)
    if args.output:
        # Exclusive creation protects existing captures and source trajectories.
        with args.output.open("xb") as output:
            output.write(raw)
    else:
        print(raw.decode("utf-8"))
