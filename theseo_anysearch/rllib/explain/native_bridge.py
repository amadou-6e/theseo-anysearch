"""Persistent JSON-lines bridge used by the native explainability UI."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from theseo_anysearch.rllib.explain.service import resolve_run_dir
from theseo_anysearch.rllib.explain.ui.session import InteractiveExplanationSession


def _json_observation(observation: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """Convert an observation into the flat JSON representation edited by Rust."""

    return {
        name: np.asarray(value, dtype=np.float32).reshape(-1).tolist()
        for name, value in observation.items()
    }


def _schema(session: InteractiveExplanationSession) -> dict[str, Any]:
    """Return editable field bounds and shapes from the checkpoint's environment."""

    fields: dict[str, Any] = {}
    for name, space in session.observation_space.spaces.items():
        fields[name] = {
            "shape": list(space.shape),
            "low": np.asarray(space.low, dtype=np.float32).reshape(-1).tolist(),
            "high": np.asarray(space.high, dtype=np.float32).reshape(-1).tolist(),
        }
    return fields


def _flatten_observation(
    raw: dict[str, Any],
    fields: dict[str, Any],
) -> dict[str, list[float]]:
    """Validate names and flatten an imported observation mapping."""

    expected = set(fields)
    received = set(raw)
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        raise ValueError(
            "imported observation fields do not match the policy schema; "
            f"missing={missing}, extra={extra}"
        )
    result: dict[str, list[float]] = {}
    for name, value in raw.items():
        array = np.asarray(value, dtype=np.float32)
        expected_shape = tuple(fields[name]["shape"])
        if array.shape != expected_shape and array.size != int(np.prod(expected_shape)):
            raise ValueError(
                f"field {name!r} has shape {array.shape}; expected {expected_shape}"
            )
        result[name] = array.reshape(-1).tolist()
    return result


def _single_tensor_observation(
    tensor: Any,
    fields: dict[str, Any],
) -> dict[str, Any]:
    """Assign a single tensor only when its matching schema field is unambiguous."""

    array = np.asarray(tensor)
    matches = [
        name for name, schema in fields.items()
        if int(np.prod(schema["shape"])) == array.size
    ]
    if len(matches) != 1:
        raise ValueError(
            "a single tensor can only be imported when exactly one observation field "
            f"matches its size; matches={matches}"
        )
    return {matches[0]: array.reshape(tuple(fields[matches[0]]["shape"]))}


def _load_observation_file(
    path: Path,
    fields: dict[str, Any],
) -> tuple[dict[str, list[float]], str]:
    """Load one fictional observation and report its automatically detected format."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = _single_tensor_observation(raw, fields)
        return _flatten_observation(raw, fields), "JSON"

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
        return _flatten_observation(raw, fields), "NumPy NPZ"

    if suffix == ".npy":
        raw = _single_tensor_observation(np.load(path, allow_pickle=False), fields)
        return _flatten_observation(raw, fields), "NumPy NPY"

    if suffix in {".pb", ".tensor"}:
        import tensorflow as tf

        proto = tf.make_tensor_proto(0)
        proto.ParseFromString(path.read_bytes())
        raw = _single_tensor_observation(tf.make_ndarray(proto), fields)
        return _flatten_observation(raw, fields), "TensorFlow TensorProto"

    raise ValueError(
        f"unsupported observation file {path.name!r}; expected JSON, NPY, NPZ, PB, or TENSOR"
    )


def _respond(payload: dict[str, Any]) -> None:
    """Write one protocol response without buffering it."""

    print(json.dumps(payload, allow_nan=False), flush=True)


def serve(run_ref: str, checkpoint: str) -> None:
    """Restore one policy and serve explanation requests until stdin closes."""

    with contextlib.redirect_stdout(sys.stderr):
        session = InteractiveExplanationSession(resolve_run_dir(run_ref), checkpoint)
        observation = session.initial_observation()
        schema = _schema(session)
    _respond(
        {
            "ok": True,
            "type": "ready",
            "observation": _json_observation(observation),
            "fields": schema,
        }
    )
    for raw_line in sys.stdin:
        request = json.loads(raw_line)
        command = request.get("command")
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if command == "explain_observation":
                    report = session.explain(request["observation"])
                elif command == "explain_trajectory":
                    report = session.service.explain_trace(
                        request["trajectory"],
                        focus="explicit",
                        explicit_steps=(int(request["step"]),),
                        max_steps=1,
                    )
                elif command == "reset_observation":
                    observation = session.initial_observation(request.get("seed"))
                    _respond({"ok": True, "observation": _json_observation(observation)})
                    continue
                elif command == "load_observation_file":
                    observation, detected_format = _load_observation_file(
                        Path(request["path"]),
                        schema,
                    )
                    _respond(
                        {
                            "ok": True,
                            "observation": observation,
                            "format": detected_format,
                        }
                    )
                    continue
                else:
                    raise ValueError(f"unsupported native explanation command: {command!r}")
            _respond({"ok": True, "report": report.to_json_dict()})
        except Exception as error:
            _respond(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )


def main() -> None:
    """Parse bridge arguments and start the request loop."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default="latest")
    arguments = parser.parse_args()
    try:
        serve(arguments.run, arguments.checkpoint)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        _respond(
            {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise


if __name__ == "__main__":
    main()
