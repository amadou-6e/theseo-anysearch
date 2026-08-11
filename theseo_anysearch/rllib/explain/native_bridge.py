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

from theseo_anysearch.rllib.explain.scenarios import validate_observation
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
    observation_space: Any,
) -> dict[str, list[float]]:
    """Reshape an imported observation to the policy schema and validate it.

    Delegates finiteness and bounds checking to the same validate_observation
    used by every other fictional-observation entry point, instead of only
    checking that the imported array's total element count matches — a
    same-size-but-wrong-shape or out-of-bounds import must fail loudly here
    too, not just when a later explain request happens to trip over it.
    """

    expected = set(fields)
    received = set(raw)
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        raise ValueError(
            "imported observation fields do not match the policy schema; "
            f"missing={missing}, extra={extra}"
        )
    reshaped: dict[str, list[float]] = {}
    for name, value in raw.items():
        array = np.asarray(value, dtype=np.float32)
        expected_shape = tuple(fields[name]["shape"])
        expected_size = int(np.prod(expected_shape))
        if array.size != expected_size:
            raise ValueError(
                f"field {name!r} has {array.size} value(s); expected {expected_size}"
            )
        reshaped[name] = array.reshape(expected_shape).tolist()
    validated = validate_observation(reshaped, observation_space)
    return {name: np.asarray(value).reshape(-1).tolist() for name, value in validated.items()}


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
    observation_space: Any,
) -> tuple[dict[str, list[float]], str]:
    """Load one fictional observation and report its automatically detected format."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = _single_tensor_observation(raw, fields)
        return _flatten_observation(raw, fields, observation_space), "JSON"

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
        return _flatten_observation(raw, fields, observation_space), "NumPy NPZ"

    if suffix == ".npy":
        raw = _single_tensor_observation(np.load(path, allow_pickle=False), fields)
        return _flatten_observation(raw, fields, observation_space), "NumPy NPY"

    if suffix in {".pb", ".tensor"}:
        import tensorflow as tf

        proto = tf.make_tensor_proto(0)
        proto.ParseFromString(path.read_bytes())
        raw = _single_tensor_observation(tf.make_ndarray(proto), fields)
        return _flatten_observation(raw, fields, observation_space), "TensorFlow TensorProto"

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
        try:
            request = json.loads(raw_line)
            command = request.get("command")
            with contextlib.redirect_stdout(sys.stderr):
                if command == "explain_observation":
                    report = session.explain(request["observation"])
                    response = {"ok": True, "report": report.to_json_dict()}
                elif command == "explain_trajectory":
                    report = session.service.explain_trace(
                        request["trajectory"],
                        focus="explicit",
                        explicit_steps=(int(request["step"]),),
                        max_steps=1,
                    )
                    response = {"ok": True, "report": report.to_json_dict()}
                elif command == "reset_observation":
                    observation = session.initial_observation(request.get("seed"))
                    response = {"ok": True, "observation": _json_observation(observation)}
                elif command == "load_observation_file":
                    observation, detected_format = _load_observation_file(
                        Path(request["path"]),
                        schema,
                        session.observation_space,
                    )
                    response = {
                        "ok": True,
                        "observation": observation,
                        "format": detected_format,
                    }
                else:
                    raise ValueError(f"unsupported native explanation command: {command!r}")
            _respond(response)
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
