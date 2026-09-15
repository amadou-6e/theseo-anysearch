"""Lossless benchmark codecs with common metadata and typed variable events.

The fixed-width candidate uses a protobuf variable-event sidecar, avoiding a
second handwritten nested-event schema. Binary layouts are little endian.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

COMMON = (
    ("step", "sint64", 0), ("action", "sint32", 0), ("reward", "double", 0.0),
    ("done", "bool", False), ("cursor_x", "uint32", 0),
    ("cursor_y", "uint32", 0), ("cursor_z", "uint32", 0),
    ("voxel_count", "uint64", 0), ("placed", "bool", False),
)
MUTATION = (("occupied", "bool", False), ("kind", "uint32", 0),
            ("active", "bool", False), ("reward_weight", "double", 0.0))
VARIABLE = ("mutations", "actions", "rewards", "cursors", "placed_per_agent", "dones")
RECORD = struct.Struct("<Hqid?IIIQ?")
OFFSET = struct.Struct("<Q")
HEADER = struct.Struct("<4sIQ")
MAGIC = b"TSB1"


def json_bytes(value, *, pretty=False):
    return json.dumps(value, allow_nan=False, ensure_ascii=True,
                      indent=2 if pretty else None,
                      separators=None if pretty else (",", ":")).encode("utf-8")


def _messages():
    """Build the checked-in schema without requiring protoc at runtime."""
    file = descriptor_pb2.FileDescriptorProto(name="trajectory_bench.proto",
                                             package="trajectory_bench", syntax="proto3")
    types = descriptor_pb2.FieldDescriptorProto
    def message(name):
        return file.message_type.add(name=name)
    def field(msg, name, number, kind, repeated=False, target=None):
        item = msg.field.add(name=name, number=number,
                             type=getattr(types, "TYPE_" + kind.upper()),
                             label=types.LABEL_REPEATED if repeated else types.LABEL_OPTIONAL)
        if target:
            item.type_name = ".trajectory_bench." + target
    coord = message("Coord")
    field(coord, "values", 1, "uint32", True)
    mutation = message("Mutation")
    for i, (name, kind, _) in enumerate(MUTATION, 1):
        field(mutation, name, i, kind)
    field(mutation, "coordinate", 5, "uint32", True)
    field(mutation, "presence", 15, "uint32")
    field(mutation, "extras_json", 16, "bytes")
    step = message("Step")
    for i, (name, kind, _) in enumerate(COMMON, 1):
        field(step, name, i, kind)
    field(step, "presence", 15, "uint32")
    field(step, "extras_json", 16, "bytes")
    field(step, "mutations", 20, "message", True, "Mutation")
    field(step, "actions", 21, "sint32", True)
    field(step, "rewards", 22, "double", True)
    field(step, "cursors", 23, "message", True, "Coord")
    field(step, "placed_per_agent", 24, "bool", True)
    field(step, "dones", 25, "bool", True)
    trajectory = message("Trajectory")
    field(trajectory, "metadata_json", 1, "bytes")
    field(trajectory, "steps", 2, "message", True, "Step")
    field(trajectory, "version", 3, "uint32")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return tuple(message_factory.GetMessageClass(pool.FindMessageTypeByName(
        "trajectory_bench." + name)) for name in ("Step", "Trajectory"))


StepMessage, TrajectoryMessage = _messages()


def encode_step(value):
    doubles = [value.get("reward", 0.0), *value.get("rewards", [])]
    doubles.extend(event.get("reward_weight", 0.0) for event in value.get("mutations", []))
    if not all(math.isfinite(number) for number in doubles):
        raise ValueError("nonfinite trajectory numeric field")
    msg = StepMessage()
    for i, (name, _, _) in enumerate(COMMON):
        if name in value:
            setattr(msg, name, value[name])
            msg.presence |= 1 << i
    for i, name in enumerate(VARIABLE, len(COMMON)):
        if name not in value:
            continue
        msg.presence |= 1 << i
        if name == "mutations":
            for event in value[name]:
                item = msg.mutations.add()
                for j, (key, _, _) in enumerate(MUTATION):
                    if key in event:
                        setattr(item, key, event[key])
                        item.presence |= 1 << j
                if "coordinate" in event:
                    item.coordinate.extend(event["coordinate"])
                    item.presence |= 1 << len(MUTATION)
                extras = {k: v for k, v in event.items()
                          if k not in {x[0] for x in MUTATION} | {"coordinate"}}
                if extras:
                    item.extras_json = json_bytes(extras)
        elif name == "cursors":
            for cursor in value[name]:
                msg.cursors.add().values.extend(cursor)
        else:
            getattr(msg, name).extend(value[name])
    extras = {k: v for k, v in value.items()
              if k not in {x[0] for x in COMMON} | set(VARIABLE)}
    if extras:
        msg.extras_json = json_bytes(extras)
    return msg


def decode_step(msg):
    value = json.loads(msg.extras_json) if msg.extras_json else {}
    for i, (name, _, _) in enumerate(COMMON):
        if msg.presence & (1 << i):
            value[name] = getattr(msg, name)
    for i, name in enumerate(VARIABLE, len(COMMON)):
        if not msg.presence & (1 << i):
            continue
        if name == "mutations":
            events = []
            for item in msg.mutations:
                event = json.loads(item.extras_json) if item.extras_json else {}
                for j, (key, _, _) in enumerate(MUTATION):
                    if item.presence & (1 << j):
                        event[key] = getattr(item, key)
                if item.presence & (1 << len(MUTATION)):
                    event["coordinate"] = list(item.coordinate)
                events.append(event)
            value[name] = events
        elif name == "cursors":
            value[name] = [list(cursor.values) for cursor in msg.cursors]
        else:
            value[name] = list(getattr(msg, name))
    return value


def split(payload):
    metadata = dict(payload)
    metadata["episode"] = dict(payload["episode"])
    steps = metadata["episode"].pop("steps")
    return metadata, steps


def join(metadata, steps):
    result = dict(metadata)
    result["episode"] = dict(metadata["episode"], steps=steps)
    return result


@dataclass
class Loaded:
    metadata: dict
    steps: object
    kind: str
    events: bytes = b""
    offsets: tuple = ()

    def __len__(self):
        return len(self.steps) // RECORD.size if self.kind == "binary" else len(self.steps)

    def step(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.kind == "json":
            return self.steps[index]
        if self.kind == "protobuf":
            return decode_step(self.steps[index])
        values = RECORD.unpack_from(self.steps, index * RECORD.size)
        mask = values[0]
        msg = StepMessage.FromString(self.events[self.offsets[index]:self.offsets[index + 1]])
        result = decode_step(msg)
        for i, (name, _, _) in enumerate(COMMON):
            if mask & (1 << i):
                result[name] = values[i + 1]
        return result

    def export(self):
        return join(self.metadata, [self.step(i) for i in range(len(self))])

    def overlay(self, index):
        """Same cumulative coordinate/occupancy resolution as Rust replay."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        result = {}
        for i in range(index + 1):
            for mutation in self.step(i).get("mutations", []):
                result[tuple(mutation["coordinate"])] = mutation["occupied"]
        return result


FORMATS = ("json-pretty", "json-zstd", "protobuf", "protobuf-zstd", "binary", "binary-zstd")


def _compress(value):
    import zstandard
    return zstandard.ZstdCompressor(level=3).compress(value)


def _decompress(value):
    import zstandard
    return zstandard.ZstdDecompressor().decompress(value)


def write(directory: Path, kind: str, payload: dict):
    if kind not in FORMATS:
        raise ValueError(kind)
    directory.mkdir(parents=True, exist_ok=True)
    metadata, steps = split(payload)
    if kind.startswith("json"):
        data = json_bytes(payload, pretty=kind == "json-pretty")
        (directory / "trajectory.data").write_bytes(_compress(data) if kind.endswith("zstd") else data)
        return
    if kind.startswith("protobuf"):
        msg = TrajectoryMessage(metadata_json=json_bytes(metadata), version=1)
        for step in steps:
            msg.steps.add().CopyFrom(encode_step(step))
        data = msg.SerializeToString(deterministic=True)
        (directory / "trajectory.data").write_bytes(_compress(data) if kind.endswith("zstd") else data)
        return
    records = bytearray(HEADER.pack(MAGIC, 1, len(steps)))
    events = bytearray()
    offsets = [0]
    for step in steps:
        if not math.isfinite(step.get("reward", 0.0)):
            raise ValueError("nonfinite trajectory reward")
        mask = sum(1 << i for i, (name, _, _) in enumerate(COMMON) if name in step)
        records.extend(RECORD.pack(mask, *(step.get(name, default) for name, _, default in COMMON)))
        event = {k: v for k, v in step.items() if k not in {x[0] for x in COMMON}}
        events.extend(encode_step(event).SerializeToString(deterministic=True))
        offsets.append(len(events))
    manifest = {"benchmark_version": 1, "metadata": metadata,
                "record_bytes": RECORD.size, "step_count": len(steps),
                "compression": "zstd-3" if kind.endswith("zstd") else "none"}
    (directory / "manifest.json").write_bytes(json_bytes(manifest, pretty=True))
    for name, data in (("steps.bin", records), ("events.pb", events),
                       ("offsets.bin", b"".join(OFFSET.pack(x) for x in offsets))):
        (directory / name).write_bytes(_compress(data) if kind.endswith("zstd") else data)


def read(directory: Path, kind: str):
    if kind not in FORMATS:
        raise ValueError(kind)
    def data(name):
        value = (directory / name).read_bytes()
        return _decompress(value) if kind.endswith("zstd") else value
    if kind.startswith("json"):
        metadata, steps = split(json.loads(data("trajectory.data")))
        return Loaded(metadata, steps, "json")
    if kind.startswith("protobuf"):
        msg = TrajectoryMessage.FromString(data("trajectory.data"))
        if msg.version != 1:
            raise ValueError("unsupported protobuf benchmark version")
        return Loaded(json.loads(msg.metadata_json), msg.steps, "protobuf")
    manifest = json.loads((directory / "manifest.json").read_bytes())
    records, events, index = data("steps.bin"), data("events.pb"), data("offsets.bin")
    magic, version, count = HEADER.unpack_from(records)
    if (magic != MAGIC or version != 1 or manifest["benchmark_version"] != 1
            or manifest["record_bytes"] != RECORD.size or manifest["step_count"] != count):
        raise ValueError("invalid binary benchmark header")
    if len(records) != HEADER.size + count * RECORD.size or len(index) != (count + 1) * OFFSET.size:
        raise ValueError("truncated binary records/index")
    offsets = tuple(x[0] for x in OFFSET.iter_unpack(index))
    if offsets[0] != 0 or offsets[-1] != len(events) or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("invalid event offsets")
    return Loaded(manifest["metadata"], records[HEADER.size:], "binary", events, offsets)
