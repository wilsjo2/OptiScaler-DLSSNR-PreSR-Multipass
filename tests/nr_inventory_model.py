"""Read-only, CPU-only map of an NR weight resource.

The PE envelope/record sizes are independently parsed. Tensor-layout hypotheses
come from the supplied mauri870/dlssnr-pytorch checkout and are checked
against every local record size and FP8 matrix. No DLL is loaded or executed.
No weight values are written; output contains offsets, shapes, hashes and stats.
"""
import argparse
import ast
import collections
import dataclasses
import hashlib
import json
import pathlib
import re
import struct
import sys
import types

import numpy as np
import pefile

RESOURCES = {"ht": "WEIGHTS_HT", "small": "WEIGHTS_CC_SMALL_FP8"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_reference_definitions(extract_path, plan_path):
    # Explicit AST allowlist excludes imports, file/network I/O, CLI, torch and
    # safetensors code. The selected routines operate only on in-memory arrays.
    names = {"Part", "_Layout", "_swin_layout", "_body_32", "_pre_block_layout",
             "_post_block_layout", "_split_layouts", "_vit_layouts", "_layouts_by_size",
             "unswizzle_matrix", "unpack_qkv3", "unpack_proj16"}
    parsed = ast.parse(extract_path.read_text(encoding="utf-8"))
    selected = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(selected) == len(names)
    mod = types.ModuleType("nr_reference_layout")
    mod.__dict__.update(dataclass=dataclasses.dataclass, np=np)
    sys.modules[mod.__name__] = mod
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(extract_path), "exec"), mod.__dict__)
    # The entire small plan module was inspected: stdlib dataclasses/typing and
    # pure geometry/table construction only. It does not load the NR runtime.
    plan = types.ModuleType("nr_reference_plan")
    sys.modules[plan.__name__] = plan
    exec(compile(plan_path.read_text(encoding="utf-8"), str(plan_path), "exec"), plan.__dict__)
    return mod, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll", type=pathlib.Path)
    parser.add_argument("--reference", type=pathlib.Path, required=True,
                        help="Reviewed dlssnr-pytorch checkout; its extract.py and plan.py hashes are recorded")
    parser.add_argument("--variant", choices=RESOURCES, default="ht")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    extract_path = args.reference / "dlssnr/extract.py"
    plan_path = args.reference / "dlssnr/plan.py"
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory")
    data = args.dll.read_bytes()
    pe = pefile.PE(data=data)
    found = []
    for kind in pe.DIRECTORY_ENTRY_RESOURCE.entries:
        for name in kind.directory.entries:
            if str(name.name) == RESOURCES[args.variant]:
                for lang in name.directory.entries:
                    s = lang.data.struct
                    found.append((pe.get_offset_from_rva(s.OffsetToData), s.Size))
    assert len(found) == 1
    base, size = found[0]
    q = lambda p: struct.unpack_from("<Q", data, p)[0]
    assert q(base) == size
    cursor = base + 8
    records = []
    while cursor < base + size:
        record_offset = cursor
        name_length = q(cursor)
        cursor += 8
        assert 0 < name_length < 128
        name = data[cursor:cursor + name_length].decode("ascii")
        cursor += name_length
        body_size = q(cursor)
        cursor += 8
        body = cursor
        assert q(body) == body_size
        payload_size = q(body + 8)
        device, = struct.unpack_from("<I", data, body + 16)
        payload_offset = body + 20
        footer = payload_offset + payload_size
        type_id, layout_id, ndim = struct.unpack_from("<IIQ", data, footer)
        dims = struct.unpack_from("<" + "I" * ndim, data, footer + 16)
        assert footer + 16 + 4 * ndim == body + body_size
        records.append(dict(name=name, record_offset=record_offset, payload_offset=payload_offset,
                            payload_bytes=payload_size, device=device, type_id=type_id,
                            layout_id=layout_id, dimensions=list(dims)))
        cursor += body_size
    assert cursor == base + size
    reference, plan = load_reference_definitions(extract_path, plan_path)
    layouts = reference._layouts_by_size()
    rows = {(r.block, r.layer): r for r in plan.ROWS[args.variant]}
    last_block = max(b for b, _ in rows)
    layers, tensors, scalars = [], [], []
    names_found, zero_matrices, nan_codes = set(), [], 0
    for record in records:
        name = record["name"]
        payload = np.frombuffer(data, np.uint8, count=record["payload_bytes"], offset=record["payload_offset"])
        match = re.fullmatch(r"block(\d+)\.layer(\d+)\.layer", name)
        if not match:
            assert name == f"block{last_block}.layer0.blend_scale" and len(payload) == 2
            scalars.append({**record, "float16_value": float(payload.view(np.float16)[0])})
            continue
        key = tuple(map(int, match.groups()))
        names_found.add(key)
        row = rows[key]
        packed_bytes = bytes(payload)
        item = {**record, "block": key[0], "layer": key[1], "reference_class": row.cls,
                "reference_kernel": row.name, "channels_in": row.c_in, "channels_out": row.c_out,
                "reference_skip": list(row.skip) if row.skip else None,
                "payload_sha256": sha(packed_bytes)}
        kernel_hits = [m.start() for m in re.finditer(re.escape(row.name.encode()), data[:base])]
        item["kernel_name_local_offsets"] = kernel_hits
        if len(payload) == 2:
            assert row.cls == "CCVit1DAttention"
            item["float16_scale"] = float(payload.view(np.float16)[0])
            item["layout_match"] = "attention scalar"
            item["tensors"] = [name.removesuffix(".layer") + ".scale"]
            tensors.append(dict(name=item["tensors"][0], file_offset=record["payload_offset"],
                                bytes=2, encoding="float16", shape=[1], value=item["float16_scale"]))
        else:
            assert len(payload) in layouts, (name, len(payload))
            family, layout = layouts[len(payload)]
            item["layout_match"] = family
            item["tensors"] = []
            for part in layout.parts:
                tname = name.removesuffix(".layer") + "." + part.name
                t = dict(name=tname, file_offset=record["payload_offset"] + part.start,
                         bytes=part.stop-part.start, encoding=part.kind, shape=list(part.shape),
                         packed_sha256=sha(packed_bytes[part.start:part.stop]))
                if part.kind in ("e4m3", "qkv3"):
                    if part.kind == "e4m3":
                        matrix = reference.unswizzle_matrix(payload, part.start, *part.shape, part.inner)
                    else:
                        matrix = reference.unpack_qkv3(payload, part.start)
                        t["shape"] = [3, 1024, 1024]
                    zeros = (matrix & 127) == 0
                    bad = int(((matrix & 127) == 127).sum())
                    nan_codes += bad
                    t.update(zero_elements=int(zeros.sum()), total_elements=matrix.size,
                             decoded_sha256=sha(matrix.tobytes()),
                             zero_fraction=float(zeros.mean()), nan_codes=bad,
                             entirely_zero_rows=int(zeros.all(axis=-1).sum()),
                             entirely_zero_columns=int(zeros.all(axis=-2).sum()))
                    if zeros.all():
                        zero_matrices.append(tname)
                item["tensors"].append(tname)
                tensors.append(t)
        layers.append(item)
    assert names_found == set(rows), (names_found ^ set(rows))
    layers.sort(key=lambda x: (x["block"], x["layer"]))
    grouped = collections.defaultdict(lambda: dict(layers=0, bytes=0))
    for x in layers:
        g = grouped[x["reference_class"]]
        g["layers"] += 1
        g["bytes"] += x["payload_bytes"]
    matrix_parts = [x for x in tensors if x["encoding"] in ("e4m3", "qkv3")]
    summary = dict(dll_path=str(args.dll.resolve()), dll_sha256=sha(data), dll_bytes=len(data),
                   variant=args.variant, resource_name=RESOURCES[args.variant],
                   resource_offset=base, resource_bytes=size, resource_sha256=sha(data[base:base+size]),
                   record_count=len(records), layer_count=len(layers), block_count=len({x["block"] for x in layers}),
                   payload_bytes=sum(x["payload_bytes"] for x in records), tensor_region_count=len(tensors),
                   matrix_region_count=len(matrix_parts), matrix_elements=sum(x["total_elements"] for x in matrix_parts),
                   matrix_zero_elements=sum(x["zero_elements"] for x in matrix_parts),
                   allzero_matrices=zero_matrices, fp8_nan_codes=nan_codes,
                   kernel_names_found=sum(bool(x["kernel_name_local_offsets"]) for x in layers),
                   family_summary=dict(grouped), additional_scalars=scalars,
                   reference_extract_sha256=sha(extract_path.read_bytes()), reference_plan_sha256=sha(plan_path.read_bytes()),
                   evidence_boundary="Record names, bytes, offsets, hashes, scalar values, and embedded kernel names are local observations. Internal tensor names/shapes and graph topology use public reverse-engineering reference layouts, checked against all local sizes; execution and activation geometry were not tested.")
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, obj in [("model-summary.json", summary), ("weight-records.json", records),
                          ("layer-map.json", layers), ("tensor-map.json", tensors)]:
        (args.output/filename).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
