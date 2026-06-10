#!/usr/bin/env python3
"""Inspect a recorded ROS2 bag — list topics + each topic's full field schema
with data type and size for every field.

Run with the system Python after sourcing ROS2::

    source /opt/ros/humble/setup.bash
    python3 deploy/robots/g1_23dof_pingpong/inspect_bag.py

Optional flags::

    --bag <dir>     bag directory (must contain metadata.yaml + *.db3)
    --topic <name>  inspect a single topic only
    --no-sample     don't peek at the first message for actual array sizes
                    (faster, but bounded/unbounded sequences just show their
                    declared length, not what the data actually holds)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_BAG = THIS_DIR / "bags" / "pingpong_sim_record"


# Bytes per element for ROS2 primitive types (CDR encoding, fixed-width).
# ROS2 idl uses 'float64' but rosidl_runtime returns the C alias 'double'
# in get_fields_and_field_types() — accept both spellings.
PRIMITIVE_SIZE = {
    "bool": 1, "boolean": 1, "byte": 1, "char": 1, "octet": 1,
    "int8": 1, "uint8": 1,
    "int16": 2, "uint16": 2,
    "int32": 4, "uint32": 4, "float32": 4, "float": 4,
    "int64": 8, "uint64": 8, "float64": 8, "double": 8, "long double": 16,
    # 'string' / 'wstring' are variable-length: handled separately
}


def _ensure_ros2_sourced():
    try:
        import rosbag2_py  # noqa: F401
        from rosidl_runtime_py.utilities import get_message  # noqa: F401
        return True
    except ImportError as exc:
        sys.exit(
            f"[inspect_bag] {exc}\n"
            "  ROS2 isn't sourced into this Python.\n"
            "  Run first:  source /opt/ros/humble/setup.bash\n"
            "  Then:       python3 deploy/robots/g1_23dof_pingpong/inspect_bag.py"
        )


def _open_bag(bag_dir: Path):
    """Open a SQLite3 ROS2 bag and return (reader, topic_metadata_list)."""
    import rosbag2_py
    storage = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)
    return reader, reader.get_all_topics_and_types()


def _msg_count_per_topic(bag_dir: Path) -> dict[str, int]:
    """Read counts from metadata.yaml so we don't iterate the whole bag."""
    import yaml
    meta_path = bag_dir / "metadata.yaml"
    if not meta_path.exists():
        return {}
    info = yaml.safe_load(meta_path.read_text())["rosbag2_bagfile_information"]
    return {
        t["topic_metadata"]["name"]: int(t["message_count"])
        for t in info.get("topics_with_message_count", [])
    }


# ──────────────────────────────────────────────────────────────────────────────
# Schema printing
# ──────────────────────────────────────────────────────────────────────────────

# Pattern matches: float64[36], string[<=10], sequence<int32>, sequence<int32, 5>
_ARRAY_RE = re.compile(
    r"""^
    (?P<elem>[\w/]+(?:\s*<[^>]+>)?)         # element type (may include nested templates)
    (?:
        \[(?P<fixed_n>\d*)\]                # fixed-size array  e.g. [36] or []
        | <=(?P<bounded_n>\d+)              # bounded string    e.g. string<=20
        | $
    )
    """,
    re.VERBOSE,
)
_SEQ_RE = re.compile(r"^sequence<\s*(?P<elem>[^,>]+?)\s*(?:,\s*(?P<n>\d+)\s*)?>$")


def _parse_field_type(type_str: str) -> tuple[str, str | None, int | None]:
    """Return (element_type, kind, count) where kind ∈ {None, 'fixed', 'sequence', 'bounded_str'}."""
    type_str = type_str.strip()
    # sequence<elem> or sequence<elem, N>
    m = _SEQ_RE.match(type_str)
    if m:
        elem = m.group("elem").strip()
        n = int(m.group("n")) if m.group("n") else None
        return elem, "sequence", n
    # bounded string: string<=N
    if type_str.startswith("string<="):
        n = int(type_str[len("string<="):])
        return "string", "bounded_str", n
    # fixed array: elem[N]   or  elem[]
    bracket = type_str.find("[")
    if bracket >= 0 and type_str.endswith("]"):
        elem = type_str[:bracket].strip()
        inside = type_str[bracket + 1 : -1].strip()
        n = int(inside) if inside.isdigit() else None
        return elem, "fixed", n
    return type_str, None, None


def _byte_size_str(elem: str, count: int | None) -> str:
    """Best-effort byte estimate for primitives + 'NxK' for arrays of primitives."""
    if elem in PRIMITIVE_SIZE:
        per = PRIMITIVE_SIZE[elem]
        if count is None:
            return f"{per} B"
        return f"{count}*{per} = {count*per} B"
    if elem in ("string", "wstring"):
        if count is None:
            return "var (4 B len + N B utf-8)"
        return f"{count} elems × var-len str"
    return ""  # nested message; size depends on message


def _resolve_msg_class(typename: str):
    """typename e.g. 'nav_msgs/msg/Odometry' or 'nav_msgs/Odometry'
    → Python message class. ROS2 fields_and_field_types returns the 2-part
    form so we normalize before lookup."""
    from rosidl_runtime_py.utilities import get_message
    parts = typename.split("/")
    if len(parts) == 2:
        typename = f"{parts[0]}/msg/{parts[1]}"
    return get_message(typename)


def _is_ros_msg(typename: str) -> bool:
    """True if 'pkg/Name' or 'pkg/msg/Name' format — i.e. nested message,
    not primitive. ROS primitives never contain '/'."""
    return "/" in typename


def _print_schema(typename: str, sample, indent: int = 0, _seen: set | None = None) -> None:
    """Recursively print message schema with type + size of each leaf."""
    _seen = _seen or set()
    if typename in _seen:
        # avoid infinite loop in unlikely recursive types
        print(f"{'  ' * indent}<recursion: {typename}>")
        return
    _seen = _seen | {typename}

    cls = _resolve_msg_class(typename)
    fields = cls.get_fields_and_field_types()

    for fname, ftype in fields.items():
        elem, kind, count = _parse_field_type(ftype)
        # actual sample value for runtime size info (e.g. dynamic seq length)
        actual = getattr(sample, fname, None) if sample is not None else None
        prefix = "  " * indent + fname

        if _is_ros_msg(elem):
            # Nested message
            if kind in ("fixed", "sequence"):
                n_disp = count if count is not None else (
                    f"runtime n={len(actual)}" if hasattr(actual, "__len__") else "var"
                )
                print(f"{prefix:<48s} type={ftype}  size=[{n_disp}]")
                # Print one inner-element schema (all elements same type)
                if hasattr(actual, "__len__") and len(actual) > 0:
                    _print_schema(elem, actual[0], indent + 1, _seen)
                else:
                    _print_schema(elem, None, indent + 1, _seen)
            else:
                print(f"{prefix:<48s} type={elem}")
                _print_schema(elem, actual, indent + 1, _seen)
        else:
            # Primitive (or array of primitives, or string)
            size_str = _byte_size_str(elem, count)
            value_str = ""
            if actual is not None and count is None:
                if elem == "string":
                    s = actual if isinstance(actual, str) else str(actual)
                    value_str = f"  len={len(s)}  example={s!r}"
                else:
                    value_str = f"  example={actual}"
            elif actual is not None:
                runtime_n = len(actual) if hasattr(actual, "__len__") else None
                if runtime_n is not None:
                    if elem in PRIMITIVE_SIZE:
                        value_str = f"  runtime_size={runtime_n}*{PRIMITIVE_SIZE[elem]} = {runtime_n*PRIMITIVE_SIZE[elem]} B"
                    else:
                        value_str = f"  runtime_n={runtime_n}"

            print(f"{prefix:<48s} type={ftype:<24s} size={size_str:<22s}{value_str}")


def _peek_first_message(reader, topic: str):
    """Return the first deserialized message for `topic` (or None if empty)."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    # fresh reader so we read from t=0 (rosbag2_py readers are forward-only).
    while reader.has_next():
        t, raw, ts = reader.read_next()
        if t == topic:
            # Look up message class from topics & types list.
            for meta in reader.get_all_topics_and_types():
                if meta.name == topic:
                    msg_cls = get_message(meta.type)
                    return deserialize_message(raw, msg_cls)
            return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", default=str(DEFAULT_BAG), help=f"bag dir (default: {DEFAULT_BAG})")
    ap.add_argument("--topic", default=None, help="inspect only this topic (default: all)")
    ap.add_argument("--no-sample", action="store_true",
                    help="skip reading the first message; faster but no runtime array sizes")
    args = ap.parse_args()

    _ensure_ros2_sourced()

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_dir():
        sys.exit(f"[inspect_bag] bag dir not found: {bag_path}")
    if not (bag_path / "metadata.yaml").exists():
        sys.exit(f"[inspect_bag] not a ROS2 bag (missing metadata.yaml): {bag_path}")

    counts = _msg_count_per_topic(bag_path)

    # First open: list topics
    reader, topic_meta = _open_bag(bag_path)
    print(f"╔══ ROS2 bag: {bag_path}")
    print(f"║   topics: {len(topic_meta)}   total msgs: {sum(counts.values()) if counts else '?'}")
    print(f"╚══")
    for meta in topic_meta:
        n = counts.get(meta.name, "?")
        print(f"  • {meta.name:<28s}  {meta.type:<32s}  msgs={n}")
    print()

    targets = [m for m in topic_meta if (args.topic is None or m.name == args.topic)]
    if args.topic and not targets:
        sys.exit(f"[inspect_bag] topic not found: {args.topic}")

    for meta in targets:
        sample = None
        if not args.no_sample:
            # Re-open bag for each peek (forward-only readers can't rewind).
            r2, _ = _open_bag(bag_path)
            sample = _peek_first_message(r2, meta.name)
        print(f"─── {meta.name}   ({meta.type})   msgs={counts.get(meta.name, '?')} ───")
        try:
            _print_schema(meta.type, sample, indent=1)
        except Exception as exc:  # noqa: BLE001
            print(f"  [schema error] {exc}")
        print()


if __name__ == "__main__":
    main()
