#!/usr/bin/env python3
"""Todo 4 (rtx2030-mfg-integration): lift the production config surface into the runner's includes.

tests/Run-AmpereMfgLoader.cmd --config must exercise the shipped read/save statements and the shipped field
declarations instead of a copy that can drift, so this tool extracts them verbatim out of Config.h/Config.cpp:

  production-config-customoptional.inc  HasDefaultValue + CustomOptional, verbatim from Config.h
  production-config-fgenabled.inc       the FGEnabled declaration, verbatim from Config.h
  production-ampere-fields.inc          the four declarations ([FrameGen] External + the three Ampere keys)
  production-config-values.inc          GetBoolValue / GetIntValue, verbatim from Config.cpp
  production-config-read.inc            Config::readString / readInt / readBool, verbatim from Config.cpp
  production-ampere-read.inc            the [FrameGen] External + [DLSSG] AmpereMfg* read statements
  production-ampere-save.inc            the [FrameGen] External + [DLSSG] AmpereMfg* save statements, including
                                        whatever removal statement still shares that block

Region anchors match exact C++ tokens with formatting whitespace ignored; extracted text stays verbatim.
Missing or duplicate anchors and changed keys fail instead of silently weakening the suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

CUSTOMOPTIONAL_SIGNATURE = (
    "template <class T, HasDefaultValue defaultState = WithDefault> class CustomOptional : public std::optional<T>"
)
ENUM_SIGNATURE = "enum HasDefaultValue"

FGENABLED_LINE = "CustomOptional<bool> FGEnabled { false };"
FIELDS_START = "CustomOptional<bool> ExternalFrameGeneration { false };"
FIELDS_END = "CustomOptional<std::string, NoDefault> FGDLSSGAmpereMfgKernelImage;"
FIELDS_REQUIRED = (
    "ExternalFrameGeneration",
    "FGDLSSGAmpereMfgUnlock",
    "FGDLSSGAmpereMfgMaxFrames",
    "FGDLSSGAmpereMfgKernelImage",
)
FIELDS_FORBIDDEN = ("FGDLSSGAda",)

VALUES = (
    ("GetBoolValue", "std::string GetBoolValue(std::optional<bool> value)"),
    ("GetIntValue", "template <typename T> std::string GetIntValue(std::optional<T> value, bool getHex = false)"),
)

READS = (
    ("readString", "std::optional<std::string> Config::readString(std::string section, std::string key, bool lowercase)"),
    ("readInt", "std::optional<int> Config::readInt(std::string section, std::string key)"),
    ("readBool", "std::optional<bool> Config::readBool(std::string section, std::string key)"),
)

READ_START = 'ExternalFrameGeneration.set_from_config(readBool("FrameGen", "External"));'
READ_END = "ExternalFrameGeneration.set_from_config(true);"
READ_REQUIRED = (
    '"FrameGen", "External"',
    '"DLSSG", "AmpereMfgUnlock"',
    '"DLSSG", "AmpereMfgMaxFrames"',
    '"DLSSG", "AmpereMfgKernelImage"',
)

SAVE_START = "bool ampereUnlock = Instance()->FGDLSSGAmpereMfgUnlock.value_for_config_or(false);"
SAVE_END = (
    'ini.SetValue("DLSSG", "AmpereMfgKernelImage", '
    'Instance()->FGDLSSGAmpereMfgKernelImage.value_for_config_or("auto").c_str());'
)
SAVE_REQUIRED = (
    'ini.SetValue("FrameGen", "External"',
    'ini.SetValue("DLSSG", "AmpereMfgUnlock"',
    'ini.SetValue("DLSSG", "AmpereMfgMaxFrames"',
    'ini.SetValue("DLSSG", "AmpereMfgKernelImage"',
    "ampereUnlock",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def skip_literal_or_comment(source: str, position: int) -> int | None:
    """If a string/char literal or comment starts at `position`, return the index just past it."""
    char = source[position]
    if char in "\"'":
        position += 1
        while position < len(source) and source[position] != char:
            position += 2 if source[position] == "\\" else 1
        return position + 1
    if source[position : position + 2] == "//":
        end = source.find("\n", position)
        return len(source) if end < 0 else end
    if source[position : position + 2] == "/*":
        end = source.find("*/", position)
        return len(source) if end < 0 else end + 2
    return None


def find_body_start(source: str, index: int) -> int:
    """Return the '{' that opens the definition body, skipping the parameter list."""
    position = source.find("(", index)
    if position < 0:
        raise SystemExit("no parameter list found")
    depth = 0
    while position < len(source):
        skipped = skip_literal_or_comment(source, position)
        if skipped is not None:
            position = skipped
            continue
        if source[position] == "(":
            depth += 1
        elif source[position] == ")":
            depth -= 1
            if depth == 0:
                break
        position += 1
    return source.find("{", position)


def extract_definition(source: str, signature: str, path: str, skip_parameters: bool = True) -> str:
    """Return the definition from its `template`/signature line through its closing brace.

    `skip_parameters` walks the parameter list before looking for the body brace; a class or enum signature
    has no parameter list before its body, so those pass False.
    """
    index = source.find(signature)
    if index < 0:
        raise SystemExit(f"anchor missing in {path}: {signature}")
    if source.find(signature, index + 1) >= 0:
        raise SystemExit(f"anchor is not unique in {path}: {signature}")

    start = source.rfind("\n", 0, index) + 1
    # A preceding template<...> line (optionally with comments) belongs to the definition.
    probe = start - 1
    for _ in range(4):
        if probe <= 0:
            break
        line_start = source.rfind("\n", 0, probe) + 1
        line = source[line_start:probe].strip()
        if line.startswith("template"):
            start = line_start
            break
        if line == "" or line.startswith("//"):
            probe = line_start - 1
            continue
        break

    body_start = find_body_start(source, index) if skip_parameters else source.find("{", index)
    if body_start < 0:
        raise SystemExit(f"no body in {path}: {signature}")
    depth = 0
    position = body_start
    while position < len(source):
        skipped = skip_literal_or_comment(source, position)
        if skipped is not None:
            position = skipped
            continue
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                # A class/enum definition keeps its trailing semicolon; a function definition has none.
                end = position + 1
                tail = source[end : end + 1]
                if tail == ";":
                    end += 1
                return source[start:end]
        position += 1
    raise SystemExit(f"unterminated body in {path}: {signature}")


def extract_line(source: str, marker: str, path: str, required: str = "") -> str:
    """Return the single line that contains `marker`, trimmed."""
    hits = source.count(marker)
    if hits != 1:
        raise SystemExit(f"marker must appear exactly once in {path}: {marker} (found {hits})")
    start = source.rfind("\n", 0, source.find(marker)) + 1
    end = source.find("\n", source.find(marker))
    if end < 0:
        end = len(source)
    line = source[start:end].strip()
    if required and required not in line:
        raise SystemExit(f"line is missing {required!r} in {path}: {line}")
    return line


def marker_pattern(marker: str) -> str:
    """Match the same C++ tokens regardless of formatting whitespace."""
    return r"\s*".join(re.escape(token) for token in re.findall(r"\w+|[^\w\s]", marker))


def extract_region(
    source: str,
    start_marker: str,
    end_marker: str,
    path: str,
    required: tuple[str, ...] = (),
    forbidden: tuple[str, ...] = (),
) -> str:
    """Return the verbatim lines from the line holding `start_marker` through the line holding `end_marker`."""
    anchors = []
    for label, marker in (("start", start_marker), ("end", end_marker)):
        hits = list(re.finditer(marker_pattern(marker), source))
        if len(hits) != 1:
            raise SystemExit(f"{label} anchor must appear exactly once in {path}: {marker} (found {len(hits)})")
        anchors.append(hits[0])

    first, last = anchors
    start = source.rfind("\n", 0, first.start()) + 1
    end = source.find("\n", last.end())
    if end < 0:
        end = len(source)
    region = source[start:end]

    if last.start() < first.start():
        raise SystemExit(f"end anchor precedes the start anchor in {path}: {end_marker}")
    for needle in required:
        if not re.search(marker_pattern(needle), region):
            raise SystemExit(f"region is missing {needle!r} in {path}")
    for needle in forbidden:
        if needle in region:
            raise SystemExit(f"region must not contain {needle!r} in {path}")
    return region


def emit(out_dir: pathlib.Path, name: str, source_path: pathlib.Path, body: str, note: str = "") -> None:
    header = [
        "// GENERATED by tools/extract_mfg_config_seam.py - do not edit.",
        f"// Verbatim from {source_path.as_posix()}.",
    ]
    if note:
        header.append(f"// {note}")
    (out_dir / name).write_text("\n".join([*header, "", body, ""]), encoding="ascii", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to Config.cpp")
    parser.add_argument("--header", required=True, help="path to Config.h")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--evidence", help="write a JSON receipt here")
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    header_path = pathlib.Path(args.header)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_source = config_path.read_text(encoding="utf-8")
    header_source = header_path.read_text(encoding="utf-8")

    extracted: dict[str, str] = {}

    enum_body = extract_definition(header_source, ENUM_SIGNATURE, header_path.as_posix(), skip_parameters=False)
    optional_body = extract_definition(
        header_source, CUSTOMOPTIONAL_SIGNATURE, header_path.as_posix(), skip_parameters=False
    )
    extracted["config.h::HasDefaultValue"] = enum_body
    extracted["config.h::CustomOptional"] = optional_body
    emit(out_dir, "production-config-customoptional.inc", header_path, "\n\n".join([enum_body, optional_body]))

    fgenabled = extract_line(header_source, FGENABLED_LINE, header_path.as_posix())
    extracted["config.h::FGEnabled"] = fgenabled
    emit(out_dir, "production-config-fgenabled.inc", header_path, fgenabled)

    fields = extract_region(
        header_source, FIELDS_START, FIELDS_END, header_path.as_posix(), FIELDS_REQUIRED, FIELDS_FORBIDDEN
    )
    extracted["config.h::ampere-fields"] = fields
    emit(
        out_dir,
        "production-ampere-fields.inc",
        header_path,
        fields,
        "[FrameGen] External and the three [DLSSG] AmpereMfg* declarations, contiguous in Config.h.",
    )

    values = "\n\n".join(
        extract_definition(config_source, signature, config_path.as_posix()) for _, signature in VALUES
    )
    for label, signature in VALUES:
        extracted[f"config.cpp::{label}"] = signature
    emit(out_dir, "production-config-values.inc", config_path, values)

    reads = "\n\n".join(
        extract_definition(config_source, signature, config_path.as_posix()) for _, signature in READS
    )
    for label, signature in READS:
        extracted[f"config.cpp::{label}"] = signature
    emit(out_dir, "production-config-read.inc", config_path, reads)

    read_region = extract_region(
        config_source, READ_START, READ_END, config_path.as_posix(), READ_REQUIRED
    )
    extracted["config.cpp::ampere-read"] = read_region
    emit(out_dir, "production-ampere-read.inc", config_path, read_region)

    save_region = extract_region(
        config_source, SAVE_START, SAVE_END, config_path.as_posix(), SAVE_REQUIRED
    )
    extracted["config.cpp::ampere-save"] = save_region
    emit(
        out_dir,
        "production-ampere-save.inc",
        config_path,
        save_region,
        "The whole [FrameGen]/[DLSSG] save block head, so any statement that still removes one of the keys "
        "is inside the compiled region and the smoke can see its effect on the produced INI.",
    )

    startup_path = config_path.with_name("dllmain.cpp")
    startup_source = startup_path.read_text(encoding="utf-8")
    startup = extract_region(
        startup_source,
        "const bool externalFg = Config::Instance()->ExternalFrameGeneration.value_or_default();",
        "State::Instance().activeFgOutput = FGOutput::NoFG;",
        startup_path.as_posix(),
        ("activeFgInput", "activeFgOutput", "activeFgNvngx"),
    )
    emit(out_dir, "production-fg-startup.inc", startup_path, startup)
    extracted["dllmain.cpp::fg-startup"] = startup

    hook_path = config_path.parent / "hooks" / "Streamline_Hooks.cpp"
    hook_source = hook_path.read_text(encoding="utf-8")
    suppression = extract_definition(
        hook_source,
        "if (state.activeFgInput != FGInput::DLSSG && state.activeFgOutput == FGOutput::DLSSG)",
        hook_path.as_posix(),
    )
    emit(out_dir, "production-fg-suppression.inc", hook_path, suppression)
    extracted["Streamline_Hooks.cpp::fg-suppression"] = suppression

    receipt = {
        "tool": "tools/extract_mfg_config_seam.py",
        "schema": 1,
        "sources": {
            config_path.as_posix(): sha256_text(config_source),
            header_path.as_posix(): sha256_text(header_source),
            startup_path.as_posix(): sha256_text(startup_source),
            hook_path.as_posix(): sha256_text(hook_source),
        },
        "extracted": {
            name: {"sha256": sha256_text(text), "lines": text.count("\n") + 1} for name, text in extracted.items()
        },
        "outputs": sorted(path.name for path in out_dir.iterdir() if path.name.endswith(".inc")),
    }
    print(
        "extracted "
        + ", ".join(f"{name} sha256={digest['sha256'][:12]}" for name, digest in receipt["extracted"].items())
    )
    if args.evidence:
        evidence_path = pathlib.Path(args.evidence)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(receipt, indent=1) + "\n", encoding="ascii", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
