#!/usr/bin/env python3
"""Verify the staged dlssg_sm86 payload against its pin manifest.

Reads PIN.json and checks every listed file's sha256 and byte size against the
directory the files were staged in. With --expect-name-from-header it also reads
the loader's shipped-name constant out of AmpereMfgLoader.h and asserts it equals
PIN.json bundled_name, so the one name the loader spells cannot drift from the
pin manifest. With --expect-digest-from-header it reads the two constants the
arming ladder enforces - kPayloadExpectedBytes and kPayloadExpectedSha256 - and
asserts BOTH equal the bundled_name entry of PIN.json, so the trust anchor the
shipped loader compares modules against (AmpereMfgLoader.h, PayloadDigestMismatch)
cannot drift from the manifest either.

Exit 0 = every requested check passed; exit 1 = at least one file is missing, has
the wrong size or digest, or a header constant does not match its PIN.json value.

Usage:
    python tools/check_payload_pin.py --pin vendor/dlssg_sm86/PIN.json --dir vendor/dlssg_sm86
    python tools/check_payload_pin.py --expect-name-from-header OptiScaler/framegen/dlssg/AmpereMfgLoader.h
    python tools/check_payload_pin.py --expect-digest-from-header OptiScaler/framegen/dlssg/AmpereMfgLoader.h
    python tools/check_payload_pin.py --dir vendor/dlssg_sm86 \
        --expect-name-from-header OptiScaler/framegen/dlssg/AmpereMfgLoader.h \
        --expect-digest-from-header OptiScaler/framegen/dlssg/AmpereMfgLoader.h

--pin defaults to vendor/dlssg_sm86/PIN.json, so the header checks work with no
arguments but the header path.
"""
import argparse
import hashlib
import json
import os
import re
import sys

CHUNK = 1 << 20

DEFAULT_PIN = os.path.join("vendor", "dlssg_sm86", "PIN.json")

# The one place the loader spells the shipped module name. The regex is anchored on
# the constant's name and shape, not on its value, so renaming the file without
# updating PIN.json fails the check instead of silently passing.
HEADER_CONSTANT = re.compile(r'kPayloadModuleName\s*\[\s*\]\s*=\s*L"([^"]*)"')

# The two constants the arming ladder enforces (todo 13). Same anchoring idea: the
# name is what is matched, the value is what is compared against PIN.json.
HEADER_BYTES_CONSTANT = re.compile(r'kPayloadExpectedBytes\s*=\s*(\d+)(ULL|UL|U|LL|L)?')
HEADER_SHA256_CONSTANT = re.compile(r'kPayloadExpectedSha256\s*\[\s*\]\s*=\s*"([0-9a-fA-F]*)"')


def read_header(header_path):
    if not os.path.isfile(header_path):
        print("FAIL: loader header not found: %s" % header_path)
        return None
    try:
        with open(header_path, encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        print("FAIL: cannot read %s: %s" % (header_path, exc))
        return None


def pinned_payload_entry(pin):
    """The PIN.json files[] entry the two header constants have to match."""
    files = pin.get("files") or []
    bundled_name = pin.get("bundled_name")
    for entry in files:
        if bundled_name and entry.get("name") == bundled_name:
            return entry
    return files[0] if files else None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def check(pin, directory):
    failures = []
    files = pin.get("files") or []
    if not files:
        print("FAIL: PIN.json lists no files")
        return 1

    for entry in files:
        name = entry["name"]
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            print("FAIL %s: missing (%s)" % (name, path))
            failures.append(name)
            continue
        size = os.path.getsize(path)
        if size != entry["bytes"]:
            print("FAIL %s: size %d != pinned %d" % (name, size, entry["bytes"]))
            failures.append(name)
            continue
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            print("FAIL %s: sha256 %s != pinned %s" % (name, digest, entry["sha256"]))
            failures.append(name)
            continue
        print("OK %s bytes=%d sha256=%s" % (name, size, digest))

    bundled_name = pin.get("bundled_name")
    if bundled_name and bundled_name not in [f["name"] for f in files]:
        print("FAIL bundled_name %r is not one of the pinned files" % bundled_name)
        failures.append(bundled_name)

    if failures:
        print("FAILED: %d of %d pinned file(s) do not match: %s"
              % (len(failures), len(files), ", ".join(failures)))
        return 1

    print("OK: all %d file(s) match PIN.json (%s@%s, tag %s)"
          % (len(files), pin.get("repo"), str(pin.get("commit"))[:7], pin.get("release_tag")))
    return 0


def expect_name_from_header(pin, header_path):
    text = read_header(header_path)
    if text is None:
        return 1

    matches = HEADER_CONSTANT.findall(text)
    if len(matches) != 1:
        print("FAIL: expected exactly one kPayloadModuleName[] = L\"...\" in %s, found %d"
              % (header_path, len(matches)))
        return 1

    from_header = matches[0]
    bundled_name = pin.get("bundled_name")
    print("header %s: kPayloadModuleName=%r" % (header_path, from_header))
    print("pin %s: bundled_name=%r" % (pin.get("repo"), bundled_name))

    if from_header != bundled_name:
        print("FAIL: the loader ships %r but PIN.json bundled_name is %r"
              % (from_header, bundled_name))
        return 1

    print("OK: the loader's shipped name equals PIN.json bundled_name (%s)" % from_header)
    return 0


def expect_digest_from_header(pin, header_path):
    """Assert the ladder's two constants equal the bundled_name entry of PIN.json.

    The size constant and the digest constant are read by name out of the header and
    compared value for value, so a flipped byte in either one - the designed failure
    fixture - exits 1 naming the constant that drifted.
    """
    text = read_header(header_path)
    if text is None:
        return 1

    entry = pinned_payload_entry(pin)
    if entry is None:
        print("FAIL: PIN.json lists no files to compare the header constants against")
        return 1

    pinned_bytes = entry.get("bytes")
    pinned_digest = entry.get("sha256")
    name = entry.get("name")

    failures = 0

    size_matches = HEADER_BYTES_CONSTANT.findall(text)
    if len(size_matches) != 1:
        print("FAIL: expected exactly one kPayloadExpectedBytes = <number> in %s, found %d"
              % (header_path, len(size_matches)))
        failures += 1
    else:
        header_bytes = int(size_matches[0][0])
        print("header %s: kPayloadExpectedBytes=%d" % (header_path, header_bytes))
        print("pin %s: %s bytes=%s" % (pin.get("repo"), name, pinned_bytes))
        if header_bytes != pinned_bytes:
            print("FAIL: kPayloadExpectedBytes is %d but PIN.json pins %s bytes for %s"
                  % (header_bytes, pinned_bytes, name))
            failures += 1

    digest_matches = HEADER_SHA256_CONSTANT.findall(text)
    if len(digest_matches) != 1:
        print("FAIL: expected exactly one kPayloadExpectedSha256[] = \"...\" in %s, found %d"
              % (header_path, len(digest_matches)))
        failures += 1
    else:
        header_digest = digest_matches[0]
        print("header %s: kPayloadExpectedSha256=%s" % (header_path, header_digest))
        print("pin %s: %s sha256=%s" % (pin.get("repo"), name, pinned_digest))
        if header_digest != pinned_digest:
            print("FAIL: kPayloadExpectedSha256 is %s but PIN.json pins %s for %s"
                  % (header_digest, pinned_digest, name))
            failures += 1

    if failures:
        print("FAILED: %d header constant(s) do not match PIN.json" % failures)
        return 1

    print("OK: kPayloadExpectedBytes=%s and kPayloadExpectedSha256=%s equal PIN.json for %s"
          % (pinned_bytes, pinned_digest, name))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", default=DEFAULT_PIN, help="path to PIN.json (default: %s)" % DEFAULT_PIN)
    parser.add_argument("--dir", help="directory holding the staged payload files")
    parser.add_argument("--expect-name-from-header", dest="expect_name_from_header",
                        help="path to AmpereMfgLoader.h; its kPayloadModuleName must equal bundled_name")
    parser.add_argument("--expect-digest-from-header", dest="expect_digest_from_header",
                        help="path to AmpereMfgLoader.h; kPayloadExpectedBytes and kPayloadExpectedSha256 must "
                             "equal the bundled_name entry of PIN.json")
    args = parser.parse_args(argv)

    if not args.dir and not args.expect_name_from_header and not args.expect_digest_from_header:
        print("FAIL: nothing to check; pass --dir, --expect-name-from-header, --expect-digest-from-header, or "
              "a combination")
        return 1

    if not os.path.isfile(args.pin):
        print("FAIL: pin manifest not found: %s" % args.pin)
        return 1
    try:
        with open(args.pin, encoding="utf-8") as handle:
            pin = json.load(handle)
    except (OSError, ValueError) as exc:
        print("FAIL: cannot parse %s: %s" % (args.pin, exc))
        return 1

    status = 0

    if args.dir:
        if not os.path.isdir(args.dir):
            print("FAIL: payload directory not found: %s" % args.dir)
            return 1
        status |= check(pin, args.dir)

    if args.expect_name_from_header:
        status |= expect_name_from_header(pin, args.expect_name_from_header)

    if args.expect_digest_from_header:
        status |= expect_digest_from_header(pin, args.expect_digest_from_header)

    return 1 if status else 0


if __name__ == "__main__":
    sys.exit(main())
