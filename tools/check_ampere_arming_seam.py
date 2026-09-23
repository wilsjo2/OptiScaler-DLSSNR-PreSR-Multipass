#!/usr/bin/env python3
"""Audit the pre-slInit arming seam in OptiScaler/hooks/Streamline_Hooks.cpp (todo 6).

The Streamline init boundary is where the RTX 20/30 (SM75/SM86) payload has to be
loaded and its companion INI applied - Streamline 2.x decides "this platform does
not support DLSS-G" inside slInit, so anything armed later is too late (C2). Two
things have to hold in the shipped file:

  * EXECUTED  - StreamlineHooks::hkslInit calls AmpereMfgLoader::Arm() before
                EVERY return in the function, not just the fall-through one. The
                DLSSG branch returns early (`return o_slInit(localPref, sdkVersion);`
                inside `if (activeFgInput == FGInput::DLSSG || activeFgOutput ==
                FGOutput::DLSSG)`), which is exactly the game-owned-FG mode the
                unlock runs in, so an arming call below it would never run.
  * INSTALLED - StreamlineHooks::hookInterposer installs the slInit detour
                (`DetourAttach(&(PVOID&) o_slInit, hkslInit)`) in every
                frame-generation ownership mode. The reference fork guards the
                whole function with
                    if (State::Instance().externalFrameGeneration) return;
                (wilsjo2/main:OptiScaler/hooks/Streamline_Hooks.cpp:1866-1867),
                which silently skips the install in the External FG ownership
                mode the Ampere unlock requires. That guard must not be ported,
                and no equivalent ownership-mode early return may appear before
                the DetourAttach.

Comments are stripped before the checks run, so the audit note that names the
fork's guard does not count as the guard itself.

Exit codes:
    0 = every check passed (the receipt names the cited guard lines)
    1 = at least one check failed, the file is missing, or --simulate-fork-guard
        did NOT reproduce the failure (a blind check)

Usage:
    python tools/check_ampere_arming_seam.py
    python tools/check_ampere_arming_seam.py --evidence .omo/evidence/.../hook-install-audit.json
    python tools/check_ampere_arming_seam.py --simulate-fork-guard
"""
import argparse
import json
import os
import re
import sys
import tempfile

DEFAULT_SOURCE = os.path.join("OptiScaler", "hooks", "Streamline_Hooks.cpp")

# The upstream guard this audit exists to keep out of the product. Recorded as a
# citation; the check below is written against the base file, not against this
# line, so it also catches a differently-spelled ownership guard.
FORK_GUARD_REFERENCE = "wilsjo2/main:OptiScaler/hooks/Streamline_Hooks.cpp:1866-1867"
FORK_GUARD_TEXT = "if (State::Instance().externalFrameGeneration) return;"

# Ownership markers that must never gate the install or the arming call.
OWNERSHIP_MARKERS = ("externalFrameGeneration", "ExternalFrameGeneration", "FGDLSSGAmpereMfgUnlock")

ARMING_CALL = "AmpereMfgLoader::Arm("
INSTALL_CALL = "DetourAttach(&(PVOID&) o_slInit, hkslInit)"
HK_INIT = "StreamlineHooks::hkslInit("
HK_INSTALL = "StreamlineHooks::hookInterposer("

# The hkslInit branch the arming call has to stay above.
INIT_EARLY_RETURN_GUARD = re.compile(r"activeFgInput\s*==\s*FGInput::DLSSG\s*\|\|\s*State::Instance\(\)\.activeFgOutput\s*==\s*"
                                     r"FGOutput::DLSSG|activeFgInput == FGInput::DLSSG")


def strip_comments(text):
    """Blank out comments while keeping every line (and its number) in place."""
    out = []
    in_block = False
    for line in text.splitlines():
        result = []
        i = 0
        while i < len(line):
            if in_block:
                end = line.find("*/", i)
                if end < 0:
                    i = len(line)
                    continue
                in_block = False
                i = end + 2
                continue
            if line.startswith("//", i):
                break
            if line.startswith("/*", i):
                in_block = True
                i += 2
                continue
            result.append(line[i])
            i += 1
        out.append("".join(result))
    return out


def function_body(lines, signature, search_from=0):
    """Line index range [first statement, closing brace) of the function, plus the signature line."""
    signature_line = -1
    for index in range(search_from, len(lines)):
        if signature in lines[index]:
            signature_line = index
            break
    if signature_line < 0:
        return None, None, None

    start = -1
    for index in range(signature_line, len(lines)):
        if "{" in lines[index]:
            start = index + 1
            break
    if start < 0:
        return None, None, None

    depth = 1
    for index in range(start, len(lines)):
        depth += lines[index].count("{") - lines[index].count("}")
        if depth <= 0:
            return signature_line, start, index

    return None, None, None


def find_line(lines, needle, start, end):
    for index in range(start, min(end, len(lines))):
        if needle in lines[index]:
            return index
    return -1


def guard_line(lines, start, end):
    """The line of the hkslInit early-return guard, if it is still there."""
    for index in range(start, min(end, len(lines))):
        if INIT_EARLY_RETURN_GUARD.search(lines[index]):
            return index
    return -1


def ownership_guard_lines(lines, start, end):
    """Code lines in [start, end) that use an ownership marker inside a conditional."""
    found = []
    for index in range(start, min(end, len(lines))):
        line = lines[index]
        if "if" not in line:
            continue
        if any(marker in line for marker in OWNERSHIP_MARKERS):
            found.append(index)
    return found


def audit(source_path):
    with open(source_path, encoding="utf-8", errors="replace") as handle:
        raw = handle.read()

    lines = strip_comments(raw)
    result = {"source": source_path, "checks": [], "citations": {}}
    failures = 0

    def record(name, ok, detail):
        nonlocal failures
        result["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failures += 1
        return ok

    init_signature, init_start, init_end = function_body(lines, HK_INIT)
    install_signature, install_start, install_end = function_body(lines, HK_INSTALL)

    if init_signature is None:
        record("hkslInit_found", False, "StreamlineHooks::hkslInit was not found in %s" % source_path)
        return 1, result

    if install_signature is None:
        record("hookInterposer_found", False, "StreamlineHooks::hookInterposer was not found in %s" % source_path)
        return 1, result

    result["citations"]["hkslInit_line"] = init_signature + 1
    result["citations"]["hookInterposer_line"] = install_signature + 1
    result["citations"]["fork_guard_reference"] = FORK_GUARD_REFERENCE
    result["citations"]["fork_guard_text"] = FORK_GUARD_TEXT

    # ---- EXECUTED: the arming call, before every return --------------------
    arming = find_line(lines, ARMING_CALL, init_start, init_end)

    record("arming_call_present", arming >= 0,
           "AmpereMfgLoader::Arm() at line %d" % (arming + 1) if arming >= 0 else
           "no AmpereMfgLoader::Arm() call inside hkslInit")

    if arming >= 0:
        result["citations"]["arming_call_line"] = arming + 1
        result["citations"]["arming_call_text"] = lines[arming].strip()

    returns = [index for index in range(init_start, init_end)
               if re.search(r"\breturn\s+o_slInit\(", lines[index])]
    result["citations"]["hkslInit_return_lines"] = [index + 1 for index in returns]

    if arming >= 0 and returns:
        first_return = min(returns)
        record("arming_before_every_return", arming < first_return,
               "arming line %d is before the first 'return o_slInit(' at line %d" %
               (arming + 1, first_return + 1) if arming < first_return else
               "arming line %d is AFTER the first 'return o_slInit(' at line %d: that early return would skip "
               "arming" % (arming + 1, first_return + 1))
    else:
        record("arming_before_every_return", False, "no arming call or no 'return o_slInit(' found in hkslInit")

    init_guard = guard_line(lines, init_start, init_end)

    if init_guard >= 0:
        result["citations"]["hkslInit_early_return_guard_line"] = init_guard + 1
        result["citations"]["hkslInit_early_return_guard_text"] = lines[init_guard].strip()

    if arming >= 0:
        window = [lines[index] for index in range(max(init_start, arming - 2), arming + 1)]
        gated = any("if" in line and any(marker in line for marker in OWNERSHIP_MARKERS) for line in window)
        record("arming_mode_independent", not gated,
               "the arming call is not gated on any ownership mode" if not gated else
               "the arming call sits under an ownership-mode conditional: %s" % " | ".join(l.strip() for l in window))
    else:
        record("arming_mode_independent", False, "no arming call to check")

    # ---- INSTALLED: the detour, in every mode ------------------------------
    install = find_line(lines, INSTALL_CALL, install_start, install_end)

    record("install_detour_present", install >= 0,
           "DetourAttach(o_slInit, hkslInit) at line %d" % (install + 1) if install >= 0 else
           "the slInit detour install was not found inside hookInterposer")

    if install >= 0:
        result["citations"]["install_detour_line"] = install + 1
        result["citations"]["install_detour_text"] = lines[install].strip()

    if install >= 0:
        gated = ownership_guard_lines(lines, install_start, install)
        record("install_mode_independent", not gated,
               "no ownership-mode conditional before the install" if not gated else
               "ownership-mode conditional before the install: line %d: %s" %
               (gated[0] + 1, lines[gated[0]].strip()))
        result["citations"]["install_path_guards"] = [index + 1 for index in gated]
    else:
        record("install_mode_independent", False, "no install call to check")

    # The fork's own spelling, in case it is ported in a way the marker scan misses.
    fork_guard = find_line(lines, "State::Instance().externalFrameGeneration", install_start, install_end)
    record("fork_guard_absent_from_install_path", fork_guard < 0,
           "no State::Instance().externalFrameGeneration in the install path" if fork_guard < 0 else
           "the reference fork's guard is present at line %d: %s" % (fork_guard + 1, lines[fork_guard].strip()))

    result["failures"] = failures
    return (1 if failures else 0), result


def simulate_fork_guard(source_path, evidence):
    """Inject the fork's early return into hookInterposer of a COPY and require the audit to catch it."""
    with open(source_path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    lines = text.splitlines(keepends=True)
    signature = -1
    for index, line in enumerate(lines):
        if HK_INSTALL in line:
            signature = index
            break

    if signature < 0:
        print("FAIL: hookInterposer not found in %s" % source_path)
        return 1

    body = signature
    while body < len(lines) and "{" not in lines[body]:
        body += 1

    injected = ["    if (State::Instance().externalFrameGeneration)\n", "        return;\n"]
    mutated = lines[: body + 1] + injected + lines[body + 1 :]
    injected_line = body + 2  # 1-based line of the injected `if`

    with tempfile.TemporaryDirectory(prefix="ampere-arming-seam-") as temp:
        copied = os.path.join(temp, os.path.basename(source_path))
        with open(copied, "w", encoding="utf-8") as handle:
            handle.writelines(mutated)

        status, result = audit(copied)

    caught = status == 1 and any(not check["ok"] for check in result["checks"])

    print("simulated fork guard: injected at line %d of the copy; audit exit=%d caught=%s"
          % (injected_line, status, "yes" if caught else "NO"))
    for check in result["checks"]:
        print("  %-40s %s  %s" % (check["check"], "ok  " if check["ok"] else "FAIL", check["detail"]))

    if evidence:
        with open(evidence, "w", encoding="utf-8") as handle:
            json.dump({"mode": "simulate-fork-guard",
                       "source": source_path,
                       "injected_line": injected_line,
                       "injected_text": "".join(injected),
                       "audit_exit": status,
                       "caught": caught,
                       "checks": result["checks"],
                       "citations": result["citations"]}, handle, indent=1)
            handle.write("\n")

    return 0 if caught else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="default: %s" % DEFAULT_SOURCE)
    parser.add_argument("--evidence", help="write the JSON receipt here")
    parser.add_argument("--simulate-fork-guard", action="store_true",
                        help="inject the reference fork's early return into a copy and require the audit to fail")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.source):
        print("FAIL: source not found: %s" % args.source)
        return 1

    if args.simulate_fork_guard:
        return simulate_fork_guard(args.source, args.evidence)

    status, result = audit(args.source)
    result["mode"] = "audit"

    for check in result["checks"]:
        print("%-40s %s  %s" % (check["check"], "ok  " if check["ok"] else "FAIL", check["detail"]))

    print("cited guard lines:")
    for key in sorted(result["citations"]):
        print("  %s = %s" % (key, result["citations"][key]))

    if args.evidence:
        with open(args.evidence, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")

    if status:
        print("FAIL: %d check(s) failed - the pre-slInit arming hook is not installed/executed in every "
              "ownership mode" % result["failures"])
    else:
        print("OK: the arming call runs before every return in hkslInit and the slInit detour is installed "
              "without an ownership-mode guard (base, not the fork)")

    return status


if __name__ == "__main__":
    sys.exit(main())
