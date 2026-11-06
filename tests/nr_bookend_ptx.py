"""Generate probe-only NR bookend PTX modules from the audited DLL; never loads GPU code.

Requires zstandard (the probe's ignored bookends/native/pydeps is also searched).
The original module is a JIT control: its two entry bodies are byte-for-byte native PTX.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "standalone/NrCapture/out-model-map/bookends/native"
sys.path.insert(0, str(DEFAULT_OUT / "pydeps"))
import zstandard

DLL_SHA256 = "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e"
PRE = "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"
POST = "cc_tinlayout_fused_post_block_swin_1h_32_fp8"
PRE_IMPORT = "nr_pre_from_lifted_fp16"
PRE_EXPORT = "nr_pre_export_lifted_fp16"
POST_EXPORT = "nr_post_export_wide_fp16"
POST_ANCHOR = "nr_post_from_anchor_fp16"
POST_WIDE_ONLY = "nr_post_wide_only_fp16"
PRE_PAIRS = [(544 + 24 * m + 6 * n, 547 + 24 * m + 6 * n) for m in range(4) for n in range(4)]
POST_PAIRS = [
    [(3805, 3806), (3807, 3808), (3815, 3816), (3817, 3818),
     (3829, 3830), (3831, 3832), (3835, 3836), (3837, 3838)],
    [(5392, 5393), (5394, 5395), (5402, 5403), (5404, 5405),
     (5416, 5417), (5418, 5419), (5423, 5424), (5425, 5426)],
]


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def entry(text: str, name: str) -> str:
    start = text.index(".visible .entry " + name + "(")
    opening = text.index("{", start)
    depth = 0
    for pos in range(opening, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos + 1] + "\n"
    raise ValueError("Unterminated PTX entry")


def extract(blob: bytes) -> tuple[bytes, str]:
    if sha256(blob) != DLL_SHA256:
        raise ValueError("Unsupported DLL hash; no native bookend variant generated")
    start = blob.index(b"\x50\xed\x55\xba")
    magic, version, outer, total = struct.unpack_from("<IHHQ", blob, start)
    offset = start + outer
    kind, entry_version, header, padded = struct.unpack_from("<HHIQ", blob, offset)
    sm = struct.unpack_from("<I", blob, offset + 28)[0]
    flags = struct.unpack_from("<Q", blob, offset + 40)[0]
    size = struct.unpack_from("<Q", blob, offset + 56)[0]
    if (version, outer, kind, entry_version, header, sm, flags) != (1, 16, 1, 257, 120, 120, 0x8041):
        raise ValueError("Unsupported pre/post fatbin envelope")
    payload = blob[offset + header:offset + header + padded]
    text = zstandard.ZstdDecompressor().decompress(payload, max_output_size=size)
    if len(text) != size or not text.startswith(b"\n"):
        raise ValueError("Invalid decompressed PTX")
    return blob[offset:offset + header], text.decode("ascii")


def fatbin(template: bytes, text: str) -> bytes:
    payload = text.encode("ascii")
    padded = (len(payload) + 7) & ~7
    header = bytearray(template)
    struct.pack_into("<Q", header, 8, padded)
    struct.pack_into("<I", header, 16, 0)
    struct.pack_into("<Q", header, 40, 0x41)
    struct.pack_into("<Q", header, 56, 0)
    outer = struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(header) + padded)
    return outer + header + payload + bytes(padded - len(payload))


def feature_io(parameter: str, pairs: list[tuple[int, int]], *, post_chunk: int | None = None,
               load: bool = False) -> str:
    """Convert the audited tensor-core C fragments to/from ordinary HWC FP16.

    A packed register holds adjacent channels for one pixel. Each m16n8 tile's
    two registers correspond to rows lane/4 and lane/4+8, columns 2*(lane%4).
    Native windows order their four 4x4 quadrants TL, TR, BL, BR.
    """
    post = post_chunk is not None
    pointer, extent = (184, 32) if post else (264, 240)
    lines = ["{", ".reg .b32 %bk_lane, %bk_group, %bk_channel, %bk_cols, %bk_rows;",
             ".reg .b32 %bk_xbase, %bk_ybase, %bk_x, %bk_y, %bk_row, %bk_tmp, %bk_pixel;",
             ".reg .b64 %bk_pointer, %bk_offset, %bk_address;", ".reg .pred %bk_xok, %bk_yok, %bk_valid;",
             f"ld.param.u64 %bk_pointer, [{parameter}+{pointer}];",
             f"ld.param.v2.u32 {{%bk_rows, %bk_cols}}, [{parameter}+{extent}];",
             "mov.u32 %bk_xbase, %ctaid.x;", "shl.b32 %bk_xbase, %bk_xbase, 3;",
             "mov.u32 %bk_ybase, %ctaid.y;", "shl.b32 %bk_ybase, %bk_ybase, 3;",
             "mov.u32 %bk_lane, %laneid;", "shr.u32 %bk_group, %bk_lane, 2;",
             "and.b32 %bk_channel, %bk_lane, 3;", "shl.b32 %bk_channel, %bk_channel, 1;"]
    if post:
        lines += [f"ld.param.u32 %bk_tmp, [{parameter}+40];", "add.s32 %bk_xbase, %bk_xbase, %bk_tmp;",
                  f"ld.param.u32 %bk_tmp, [{parameter}+44];", "add.s32 %bk_ybase, %bk_ybase, %bk_tmp;"]
    for tile, pair in enumerate(pairs):
        m, n = divmod(tile, 4)
        for high, register in enumerate(pair):
            # A pre window has four 4x4 M tiles; post produces two tiles per
            # chunk, with the second chunk four rows further down.
            yquad = int(post_chunk) if post else m // 2
            lines += [f"add.u32 %bk_row, %bk_group, {8 * high};",
                      "and.b32 %bk_x, %bk_row, 3;", "add.s32 %bk_x, %bk_x, %bk_xbase;",
                      f"add.s32 %bk_x, %bk_x, {4 * (m % 2)};",
                      "shr.u32 %bk_y, %bk_row, 2;", "add.s32 %bk_y, %bk_y, %bk_ybase;",
                      f"add.s32 %bk_y, %bk_y, {4 * yquad};",
                      "setp.lt.u32 %bk_xok, %bk_x, %bk_cols;", "setp.lt.u32 %bk_yok, %bk_y, %bk_rows;",
                      "and.pred %bk_valid, %bk_xok, %bk_yok;",
                      "mad.lo.u32 %bk_pixel, %bk_y, %bk_cols, %bk_x;",
                      "mul.wide.u32 %bk_offset, %bk_pixel, 64;",
                      f"add.u32 %bk_tmp, %bk_channel, {8 * n};", "shl.b32 %bk_tmp, %bk_tmp, 1;",
                      "cvt.u64.u32 %bk_address, %bk_tmp;", "add.u64 %bk_offset, %bk_offset, %bk_address;",
                      "add.u64 %bk_address, %bk_pointer, %bk_offset;"]
            if load:
                lines += [f"mov.b32 %r{register}, 0;", f"@%bk_valid ld.global.b32 %r{register}, [%bk_address];"]
            else:
                lines += [f"@%bk_valid st.global.b32 [%bk_address], %r{register};"]
    return "\n".join(lines + ["}", ""])


def check_mapping() -> None:
    def coordinates(post: bool):
        result = []
        for chunk in range(2 if post else 1):
            for tile in range(8 if post else 16):
                m, n = divmod(tile, 4)
                for high in range(2):
                    for lane in range(32):
                        row = lane // 4 + 8 * high
                        x = row % 4 + 4 * (m % 2)
                        y = row // 4 + 4 * (chunk if post else m // 2)
                        for half in range(2):
                            result.append((y, x, 8 * n + 2 * (lane % 4) + half))
        return result
    expected = {(y, x, c) for y in range(8) for x in range(8) for c in range(32)}
    for post in (False, True):
        mapping = coordinates(post)
        if len(mapping) != 2048 or set(mapping) != expected:
            raise ValueError("Fragment mapping does not cover each feature exactly once")


def variants(pre: str, post: str) -> tuple[str, str, str]:
    # Exact source hash and boundary signatures guard all transformations.
    if sha256(pre.encode()) != "2a1004dea53aa167cb110bc25ed94d758ab4511b17cdb17dea65ffe66109a031" or \
       sha256(post.encode()) != "b2471a490b1c5a8817689d14358d648283823ced2165f84b0fef3e13ee928bc8":
        raise ValueError("Native entry body hash differs from audited PTX")
    cut = pre.index("bar.sync 0;", pre.index("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r634, %r637}"))
    begin = pre.index("mov.b64 %rd16,")
    # The exact 37 values crossing this cut are 32 lifted half2 registers,
    # the zero accumulator, parameter address, skip address and weight pointers.
    preserved = (set(re.findall(r"%[a-z]+\d+", pre[:cut])) & set(re.findall(r"%[a-z]+\d+", pre[cut:])))
    expected = {f"%r{r}" for pair in PRE_PAIRS for r in pair} | {"%r4782", "%rd6", "%rd16", "%rd54", "%rd55"}
    if preserved != expected:
        raise ValueError("Unexpected live value crosses the native pre lift boundary")
    pre_parameter = PRE + "_param_0"
    initial = f"mov.b64 %rd16, {pre_parameter};\nld.param.u64 %rd6, [%rd16+216];\n" \
              "ld.param.u64 %rd54, [%rd16+224];\ncvta.to.global.u64 %rd55, %rd54;\nmov.b32 %r4782, 0;\n"
    imported = pre[:begin] + initial + feature_io(pre_parameter, PRE_PAIRS, load=True) + pre[cut:]
    # The importer no longer uses the original temporary 16-feature shared tile.
    imported = re.sub(r"(?m)^\.shared .*\n", "", imported)
    exported = pre[:cut] + feature_io(pre_parameter, PRE_PAIRS) + pre[cut:]
    imported = imported.replace(PRE, PRE_IMPORT).replace("_param_0[264]", "_param_0[272]")
    exported = exported.replace(PRE, PRE_EXPORT).replace("_param_0[264]", "_param_0[272]")
    post_export = post
    for chunk, marker in reversed(list(enumerate(("mov.u32 %r3803, %laneid;", "mov.u32 %r5390, %laneid;")))):
        # Insert before the output-projection weight loads; original arithmetic,
        # composition, texture access and surface output remain present.
        if post_export.count(marker) != 1:
            raise ValueError("Post projection insertion marker is not unique")
        pos = post_export.index(marker)
        post_export = post_export[:pos] + feature_io(POST + "_param_0", POST_PAIRS[chunk], post_chunk=chunk) + post_export[pos:]
    post_export = post_export.replace(POST, POST_EXPORT).replace("_param_0[184]", "_param_0[192]")
    return imported, exported, post_export


def anchor_variant(post: str) -> str:
    """Replace only the source-RGB texture fetches with an exact half buffer read.

    Caller must retain the original nonzero source handle and identity source
    rectangle, and supply a tightly packed valid-image RGBA16F buffer. Existing
    native bounds checks skip source fetches outside that valid image.
    """
    sites = [
        ("tex.2d.v4.f32.f32 {%r4694, %r4695, %r4696, %r4697}, [%rd4, {%r4691, %r4693}];", 4694, 4881, 4882),
        ("tex.2d.v4.f32.f32 {%r5724, %r5725, %r5726, %r5727}, [%rd4, {%r5721, %r5723}];", 5724, 5913, 5914),
    ]
    result = post
    for site, output, x, y in sites:
        if result.count(site) != 1:
            raise ValueError("Source-RGB fetch marker is not unique")
        lines = ["{", ".reg .b32 %ba_width, %ba_pixel, %ba_rg, %ba_ba;",
                 ".reg .b64 %ba_base, %ba_offset, %ba_address;", ".reg .b16 %ba_h0, %ba_h1, %ba_h2, %ba_h3;",
                 f"ld.param.u64 %ba_base, [{POST}_param_0+184];",
                 f"ld.param.u32 %ba_width, [{POST}_param_0+172];",
                 f"mad.lo.u32 %ba_pixel, %r{y}, %ba_width, %r{x};",
                 "mul.wide.u32 %ba_offset, %ba_pixel, 8;", "add.u64 %ba_address, %ba_base, %ba_offset;",
                 "ld.global.v2.b32 {%ba_rg, %ba_ba}, [%ba_address];",
                 "mov.b32 {%ba_h0, %ba_h1}, %ba_rg;", "mov.b32 {%ba_h2, %ba_h3}, %ba_ba;"]
        lines += [f"cvt.f32.f16 %r{output + i}, %ba_h{i};" for i in range(4)]
        result = result.replace(site, "\n".join(lines + ["}"]))
    return result.replace(POST, POST_ANCHOR).replace("_param_0[184]", "_param_0[192]")


def wide_only_variant(post: str) -> str:
    """Keep both post-Swin chunks, export their features, omit only RGB tails.

    The first RGB tail sits BETWEEN the two post-Swin chunks. Its only register
    names also used by the second Swin chunk are the already-defined zero and
    weight pointer; the deleted tail does not assign either value.
    """
    markers = ("mov.u32 %r3803, %laneid;", "$L__BB5_60:",
               "mov.u32 %r5390, %laneid;", "$L__BB5_72:")
    if any(post.count(marker) != 1 for marker in markers):
        raise ValueError("Wide-only post boundary is not unique")
    a, b, c, d = (post.index(marker) for marker in markers)
    if not a < b < c < d:
        raise ValueError("Wide-only post boundaries are out of order")
    first_tail, second_swin, last_tail = post[a:b], post[b:c], post[c:d]
    numeric_regs = lambda section: set(re.findall(r"%[a-z]+\d+", section))
    if numeric_regs(first_tail) & numeric_regs(second_swin) != {"%r5420", "%rd6"}:
        raise ValueError("Unexpected first RGB tail / second Swin register coupling")
    # Both shared names occur only as input operands in the removed tail.
    if re.search(r"(?m)^\w[\w.]*\s+(?:%r5420|%rd6)[,;]", first_tail):
        raise ValueError("Removed RGB tail changes a surviving Swin input")
    projection = "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
    if first_tail.count(projection) != 8 or last_tail.count(projection) != 8:
        raise ValueError("Unexpected RGB projection instruction count")
    result = post[:a] + feature_io(POST + "_param_0", POST_PAIRS[0], post_chunk=0) \
             + second_swin + feature_io(POST + "_param_0", POST_PAIRS[1], post_chunk=1) + post[d:]
    labels = set(re.findall(r"(?m)^(\$L__\w+):", result))
    targets = set(re.findall(r"\bbra(?:\.uni)?\s+(\$L__\w+)", result))
    if targets - labels or "sust." in result or "tex." in result or projection in result:
        raise ValueError("Wide-only post retained an RGB operation or dangling branch")
    return result.replace(POST, POST_WIDE_ONLY).replace("_param_0[184]", "_param_0[192]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, default=ROOT / "standalone/NrCapture/runtime/nvngx_dlssnr.dll")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    template, text = extract(args.dll.read_bytes())
    prefix = text[:text.index(".visible .entry ")]
    pre, post = entry(text, PRE), entry(text, POST)
    original = prefix + pre + "\n" + post
    check_mapping()
    imported, exported, post_export = variants(pre, post)
    anchor = anchor_variant(post)
    wide_only = wide_only_variant(post)
    features = prefix + imported + "\n" + exported + "\n" + post_export + "\n" + anchor + "\n" + wide_only
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, body in {"bookend-original": original, "bookend-features": features}.items():
        ptx = body.encode("ascii")
        binary = fatbin(template, body)
        (args.out / (name + ".ptx")).write_bytes(ptx)
        (args.out / (name + ".fatbin")).write_bytes(binary)
        outputs[name] = {"ptx_bytes": len(ptx), "ptx_sha256": sha256(ptx),
                         "fatbin_bytes": len(binary), "fatbin_sha256": sha256(binary)}
    manifest = {"schema": 1, "source_dll_sha256": DLL_SHA256, "ptx_version": "9.4", "target": "sm_120",
                "entries": {PRE: {"param_bytes": 264, "original_body_sha256": sha256(pre.encode())},
                            POST: {"param_bytes": 184, "original_body_sha256": sha256(post.encode())}},
                "feature_entries": {PRE_IMPORT: {"param_bytes": 272, "pointer_offset": 264, "operation": "read lifted features"},
                                    PRE_EXPORT: {"param_bytes": 272, "pointer_offset": 264, "operation": "write lifted features; preserve original pre output"},
                                    POST_EXPORT: {"param_bytes": 192, "pointer_offset": 184, "operation": "write post-Swin features; preserve original RGB output"},
                                    POST_ANCHOR: {"param_bytes": 192, "pointer_offset": 184, "operation": "read valid-image HWC RGBA16F source anchor; replace only two source texture fetches"},
                                    POST_WIDE_ONLY: {"param_bytes": 192, "pointer_offset": 184, "operation": "write post-Swin features, omitting both RGB projection/composition/surface-output tails; preserve both complete Swin chunks"}},
                "feature_layout": "HWC32 FP16; address = base + 2*((y*padded_columns+x)*32+channel); bytes=rows*columns*64",
                "anchor_layout": "HWC RGBA16F, valid image; address=base+8*(y*width+x); width=param172, height=param176",
                "anchor_requirements": "Nonzero original source texture handle at56, identity source rectangle/transforms64..84, original current-image width/height172/176. Padded feature extent32/36 is NOT anchor extent.",
                "outputs": outputs,
                "note": "Original control entry bodies unchanged. PTX-only fatbin forces driver JIT; no GPU execution by generator."}
    (args.out / "bookend-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
