"""Generate OptiScaler/dlssnr/NrPassBorrowTables.h from recorded NR kernel traces.

Usage (from the repository root):
    python OptiScaler/dlssnr/tools/make_pass_borrow_tables.py [--check]

Reads the NvAPI kernel traces recorded by NrModelProbe under standalone/NrCapture/out-model-map (gitignored research
output) and the two audited runtime DLLs, derives every cut's launch range, handoff pointer and byte formula, and
asserts the contracts NrPassBorrow.cpp relies on, on every trace and every cut:
  - every trace of a variant has the identical kernel name + paramSize sequence;
  - the producer's output pointer (one fixed parameter offset) equals the consumer's p0;
  - the handoff byte formula (two UINT parameters of the first skipped launch times a constant) equals rows x cols x
    channels of the cut level derived independently from the pre-block extent and the kernel names, at every
    resolution, and equals the producer's own extent;
  - the handoff region [p0, p0 + bytes) contains no other pointer carried from before the cut to after it (encoder
    skips, counters, scratch), and the runtime form of that check (consumer parameters only) passes;
  - no completion counter referenced inside the skipped range is referenced outside it (except by the arena clear);
  - the consumer has no input counter (none of its counters is referenced by an earlier launch except the clear);
  - every pointer used both inside the skipped range and after it, and never before it, is the handoff itself.
A cut that fails any check is emitted as unsupported (absent) for that variant, with the reason in the header.
--check compares the generated text with the existing header instead of writing it.
"""
import csv
import hashlib
import pathlib
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
MAP = REPO / "standalone/NrCapture/out-model-map"
OUT = REPO / "OptiScaler/dlssnr/NrPassBorrowTables.h"
NATIVE = "model-b-20261106/native/"
CINEMATIC = "cinematic-4k-20261106/raw/"
# The "Cinematic" layout is every non-Standard style: Natural (style 1) launches exactly the Cinematic (style 2) sequence
# (names, grid, block, shared, param size) for both models at 4K, natural-trace-20261107/raw/ (2026-11-07). Those
# traces are not listed below: a second same-resolution trace would trip the three-resolution rule in derive_cut.

RUNTIMES = {  # name -> (file, expected prefix, expected suffix)
    "310.8": ("standalone/NrCapture/runtime/nvngx_dlssnr.dll", "e16bcf15", "1fc8e"),
    "310.9.16": ("standalone/NrCapture/out-model-map/model-b-20261106/runtime-310.9.16/nvngx_dlssnr.dll", "2bc19bc0",
                 "789f"),
}
# (model, runtime, layout, traces)
VARIANTS = [
    ("A", "310.8", "Default", [NATIVE + f"{r}-old-a-0.rgba16f.trace.csv" for r in ("1080p", "1440p", "2160p")]),
    ("A", "310.9.16", "Default", [NATIVE + f"{r}-new-a-1.rgba16f.trace.csv" for r in ("1080p", "1440p", "2160p")]),
    ("B", "310.9.16", "Default", [NATIVE + f"{r}-new-b-2.rgba16f.trace.csv" for r in ("1080p", "1440p", "2160p")] +
     [NATIVE + n for n in ("auto-1080.rgba16f.trace.csv", "debug-b-1080.rgba16f.trace.csv",
                           "debug-b-2160.rgba16f.trace.csv")]),
    ("A", "310.9.16", "Cinematic", [CINEMATIC + "nightclub-model-a-pass-1.rgba16f.trace.csv"]),
    ("B", "310.9.16", "Cinematic", [CINEMATIC + "nightclub-model-b-pass-1.rgba16f.trace.csv"]),
]
# Default-layout skipped ranges (first, last), shallowest to deepest. Cinematic adds one leading launch.
RANGES = {
    "A": {"Transformer": (57, 98), "Below32": (24, 131), "Below16": (16, 139), "Below8": (10, 145),
          "Below4": (6, 149), "Below2": (2, 153)},
    "B": {"Transformer": (19, 60), "Below32": (10, 69), "Below16": (8, 71), "Below8": (6, 73),
          "Below4": (4, 75), "Below2": (2, 77)},
}
CUTS = ["Transformer", "Below32", "Below16", "Below8", "Below4", "Below2"]
# Channels and spatial shift (relative to the pre-block's half-resolution output) of each cut's handoff tensor.
LEVEL = {"Transformer": (1024, 5), "Below32": (512, 4), "Below16": (256, 3), "Below8": (128, 2), "Below4": (64, 1),
         "Below2": (32, 0)}
# Kernel-name token of the kernels running at each cut's level.
LEVEL_TOKEN = {"Transformer": "cc_vit_1d", "Below32": "_16h_", "Below16": "_8h_256", "Below8": "_4h_128",
               "Below4": "_2h_64", "Below2": "_1h_32"}
# Channel count as it appears in the producer (or, for the ViT, consumer) kernel name.
NAME_CHANNELS = {"Transformer": "_1024_", "Below32": "_512_", "Below16": "_256_", "Below8": "_128_", "Below4": "_64_",
                 "Below2": "_32_"}


class Fail(Exception):
    pass


def require(condition, message):
    if not condition:
        raise Fail(message)


class Trace:
    def __init__(self, path):
        self.path = path
        rows = list(csv.DictReader(open(MAP / path, newline="")))
        require(rows, f"{path}: empty")
        self.names = [r["name"] for r in rows]
        self.sizes = [int(r["param_size"]) for r in rows]
        self.grids = [(int(r["grid_x"]), int(r["grid_y"]), int(r["grid_z"])) for r in rows]
        self.params = [bytes.fromhex(r["params_hex"]) for r in rows]
        for i, (p, s) in enumerate(zip(self.params, self.sizes)):
            require(len(p) == s, f"{path}: launch {i} has {len(p)} parameter bytes, paramSize {s}")
        self.clear = self.names.index("cc_cb_clear")
        self.arena = self.q(self.clear, 0)
        self.arena_end = self.arena + self.u(self.clear, 8) * 4

    def __len__(self):
        return len(self.names)

    def q(self, i, o):
        return struct.unpack_from("<Q", self.params[i], o)[0]

    def u(self, i, o):
        return struct.unpack_from("<I", self.params[i], o)[0]

    def qwords(self, i, skip_slot=None):
        return [(o, self.q(i, o)) for o in range(0, len(self.params[i]) - 7, 8) if o != skip_slot]

    def values(self, i):
        return {v for _, v in self.qwords(i)}

    def counters(self, i):
        return {v for _, v in self.qwords(i) if self.arena <= v < self.arena_end}

    def pointers(self, i):
        return {v for _, v in self.qwords(i) if pointer(v) and not self.arena <= v < self.arena_end}


def pointer(v):
    # Arena and weight addresses are 16-byte aligned GPU VAs; packed extents, -4 offsets and floats are not.
    return v % 16 == 0 and 0x10000000 <= v < 1 << 48


def sha256(path):
    h = hashlib.sha256()
    with open(REPO / path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def derive_cut(traces, evidence, model, layout, cut, log):
    shift = 1 if layout == "Cinematic" else 0
    first, last = (v + shift for v in RANGES[model][cut])
    producer, consumer = last, last + 1
    t0 = traces[0]
    require(consumer < len(t0), "consumer past the end")
    channels, level = LEVEL[cut]
    named = t0.names[consumer if cut == "Transformer" else producer]
    require(NAME_CHANNELS[cut] in named, f"channel count {channels} not in kernel name {named}")
    # Producer output offset: the unique parameter slot equal to the consumer's p0 in every trace.
    slots = None
    for t in traces:
        hits = {o for o, v in t.qwords(producer) if v == t.q(consumer, 0)}
        slots = hits if slots is None else slots & hits
    require(slots and len(slots) == 1, f"producer output slot not unique: {sorted(slots or [])}")
    output = slots.pop()
    # Independent extent: the first C32 kernel after the pre-block carries the half-resolution padded extent.
    c32 = shift + 2
    require("swin_1h_32_1_inpview" in t0.names[c32], f"launch {c32} is not the first C32 kernel")

    def expected(t):
        rows, cols = t.u(c32, 24), t.u(c32, 28)
        require(rows and cols and rows % 32 == 0 and cols % 32 == 0, f"{t.path}: odd half-res extent {rows}x{cols}")
        # Observed padding: the 1/32 level rounds up to a multiple of 4, the ViT grid is that halved, again rounded.
        for step in range(min(level, 4)):
            rows, cols = rows >> 1, cols >> 1
        if level >= 4:
            rows, cols = (rows + 3) // 4 * 4, (cols + 3) // 4 * 4
        if level == 5:
            rows, cols = (rows // 2 + 3) // 4 * 4, (cols // 2 + 3) // 4 * 4
        return rows * cols * channels

    def pairs(t, i):
        return [o for o in range(0, len(t.params[i]) - 7, 4)
                if t.u(i, o) and t.u(i, o) * t.u(i, o + 4) * channels == expected(t)]

    dims = first  # the operands are known when skipping would start
    common = None
    for t in traces:
        hits = set(pairs(t, dims))
        common = hits if common is None else common & hits
    require(common, f"no extent pair in launch {dims} gives rows x cols x {channels}")
    rows_off = min(common)
    common = None
    for t in traces:
        hits = set(pairs(t, producer))
        common = hits if common is None else common & hits
    require(common, f"no extent pair in producer {producer}")
    check_off = min(common)
    resolutions = set()
    for t in traces:
        n = len(t)
        p0 = t.q(consumer, 0)
        rows, cols = t.u(dims, rows_off), t.u(dims, rows_off + 4)
        nbytes = rows * cols * channels
        resolutions.add((t.u(c32, 24), t.u(c32, 28)))
        require((t.u(producer, check_off), t.u(producer, check_off + 4)) == (rows, cols), f"{t.path}: producer extent")
        require(t.q(producer, output) == p0, f"{t.path}: producer output != consumer p0")
        require(p0 % 16 == 0 and nbytes % 16 == 0, f"{t.path}: handoff not 16-byte aligned")
        before = set().union(*(t.values(i) for i in range(n) if i < first and i != t.clear))
        inside = set().union(*(t.values(i) for i in range(first, last + 1)))
        after = set().union(*(t.values(i) for i in range(last + 1, n)))
        after_no_p0 = set().union(*({v for _, v in t.qwords(i, 0 if i == consumer else None)}
                                    for i in range(last + 1, n)))
        carried = [v for v in before & after_no_p0 if p0 < v < p0 + nbytes]
        require(not carried, f"{t.path}: handoff region overlaps carried pointers {[hex(v) for v in carried]}")
        runtime = [v for _, v in t.qwords(consumer, 0) if v in before and p0 <= v < p0 + nbytes]
        require(not runtime, f"{t.path}: runtime alias check fails {[hex(v) for v in runtime]}")
        inside_c = set().union(*(t.counters(i) for i in range(first, last + 1)))
        outside_c = set().union(*(t.counters(i) for i in range(n) if i != t.clear and not first <= i <= last))
        require(not inside_c & outside_c, f"{t.path}: counters shared across the cut {len(inside_c & outside_c)}")
        earlier_c = set().union(*(t.counters(i) for i in range(consumer) if i != t.clear))
        require(not t.counters(consumer) & earlier_c, f"{t.path}: consumer waits on an earlier counter")
        require(not (t.counters(consumer) & inside_c), f"{t.path}: consumer shares a skipped counter")
        # A pointer used inside the skipped range and after it (and never before) must be the handoff, or allocator
        # reuse of a dead buffer: real data flow keeps the two slots equal in every trace, while reuse depends on the
        # resolution, so each such slot pair must differ in at least one trace of the same kernels.
        ptr_inside = set().union(*(t.pointers(i) for i in range(first, last + 1)))
        ptr_after = set().union(*(t.pointers(i) for i in range(last + 1, n)))
        ptr_before = set().union(*(t.pointers(i) for i in range(first) if i != t.clear))
        reused = 0
        for v in (ptr_inside & ptr_after) - ptr_before - {p0}:
            uses_in = [(i, o) for i in range(first, last + 1) for o, x in t.qwords(i) if x == v]
            uses_after = [(i, o) for i in range(last + 1, n) for o, x in t.qwords(i) if x == v]
            for a in uses_in:
                for b in uses_after:
                    require(any(e.q(a[0] + d, a[1]) != e.q(b[0] + d, b[1]) for e, d in evidence),
                            f"{t.path}: launch {a[0]} p{a[1]} feeds launch {b[0]} p{b[1]} in every trace")
            reused += 1
        # Allocation evidence: two pointers of this level's kernels are exactly the handoff size apart.
        token = LEVEL_TOKEN[cut]
        level_ptrs = sorted(set().union(*(t.pointers(i) for i in range(n) if token in t.names[i])))
        gaps = {b - a for a, b in zip(level_ptrs, level_ptrs[1:])}
        require(nbytes in gaps, f"{t.path}: no {token} allocation spaced {nbytes} bytes apart")
        # Encoder skips / scratch the later launches take from before the cut (informational).
        skips = sorted(v for v in (after & before) - inside if v > 0x10000000 and not t.arena <= v < t.arena_end)
        log.append(f"  {cut:<11} {t.path.split('/')[-1]:<44} bytes {nbytes:>10} p0 {p0:#x} "
                   f"skips-from-before {len(skips)} reused {reused}")
    if len(traces) > 1:
        require(len(resolutions) >= 3, f"formula checked at only {len(resolutions)} resolutions")
    return dict(cut=cut, first=first, last=last, output=output, dims=dims, rows=rows_off, cols=rows_off + 4,
                check_rows=check_off, check_cols=check_off + 4, scale=channels, resolutions=len(resolutions))


def main():
    log = []
    hashes = {}
    for name, (path, prefix, suffix) in RUNTIMES.items():
        h = sha256(path)
        require(h.startswith(prefix) and h.endswith(suffix), f"{path}: unexpected hash {h}")
        hashes[name] = h
        log.append(f"runtime {name}: {h}")
    variants = []
    for model, runtime, layout, paths in VARIANTS:
        missing = [p for p in paths if not (MAP / p).exists()]
        if missing:
            log.append(f"variant {model} {runtime} {layout}: absent (missing {missing})")
            continue
        traces = [Trace(p) for p in paths]
        t0 = traces[0]
        for t in traces[1:]:
            require(t.names == t0.names and t.sizes == t0.sizes, f"{t.path}: sequence differs from {t0.path}")
        expected_len = {"A": 156, "B": 80}[model] + (2 if layout == "Cinematic" else 0)
        require(len(t0) == expected_len, f"{t0.path}: {len(t0)} launches, expected {expected_len}")
        require(t0.clear == (1 if layout == "Cinematic" else 0), f"{t0.path}: clear at {t0.clear}")
        if layout == "Cinematic":
            require(t0.names[0] == "cg2r_copy_kernel" and t0.names[-1] == "cg2r_post_process_kernel",
                    f"{t0.path}: not the Cinematic layout")
        pre = t0.sizes[t0.clear + 1]
        require(pre == {"310.8": 264, "310.9.16": 272}[runtime], f"{t0.path}: pre-block paramSize {pre}")
        log.append(f"variant {model} {runtime} {layout}: {len(t0)} launches, {len(traces)} traces")
        # Same-kernel evidence: this variant's traces, plus (for Cinematic) the Default traces shifted by one launch.
        evidence = [(t, 0) for t in traces]
        if layout == "Cinematic":
            for m, r, l, other in VARIANTS:
                if (m, r, l) == (model, runtime, "Default"):
                    for e in (Trace(p) for p in other if (MAP / p).exists()):
                        require(e.names == t0.names[1:-1], f"{e.path}: not the Default form of {t0.path}")
                        evidence.append((e, -1))
        cuts, unsupported = [], []
        for cut in CUTS:
            try:
                cuts.append(derive_cut(traces, evidence, model, layout, cut, log))
            except Fail as error:
                unsupported.append((cut, str(error)))
                log.append(f"  {cut}: UNSUPPORTED: {error}")
        variants.append(dict(model=model, runtime=runtime, layout=layout, trace=t0, traces=paths, cuts=cuts,
                             unsupported=unsupported))
    text = emit(variants, hashes)
    print("\n".join(log))
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT} is out of date")
            return 1
        print(f"{OUT} is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


def emit(variants, hashes):
    out = []
    w = out.append
    w("#pragma once")
    w("")
    w("// Generated by OptiScaler/dlssnr/tools/make_pass_borrow_tables.py from the NvAPI kernel traces under")
    w("// standalone/NrCapture/out-model-map. Do not edit; rerun the generator.")
    w("//")
    w("// Every cut below was asserted on every listed trace: producer output == consumer p0, byte formula == the")
    w("// level's rows x cols x channels at every resolution, no carried pointer inside the handoff region, no counter")
    w("// shared across the cut, no input counter on the consumer, and no pointer other than the handoff flowing from")
    w("// the skipped launches to later ones. Variants or cuts without such evidence are absent and refused.")
    w("")
    w("#include <cstdint>")
    w("")
    w("namespace NrPassBorrow::Tables")
    w("{")
    w("")
    w("enum class Runtime : uint8_t")
    w("{")
    w("    Unknown = 0,")
    w("    R310_8 = 1,    // Model A only")
    w("    R310_9_16 = 2, // Models A and B")
    w("};")
    w("")
    w("struct Audited")
    w("{")
    w("    const char* sha256; // lower-case hex")
    w("    Runtime runtime;")
    w("    const char* name;")
    w("};")
    w("inline constexpr Audited kAudited[] = {")
    for name, h in hashes.items():
        w(f'    {{ "{h}", Runtime::R{name.replace(".", "_")}, "{name}" }},')
    w("};")
    w("")
    w("struct Kernel")
    w("{")
    w("    const char* name;")
    w("    uint16_t paramSize;")
    w("};")
    w("")
    w("// Launch indices count from the first launch of one evaluation. Skipped: first..last inclusive. The producer is")
    w("// `last` (output pointer at byte offset `output`), the consumer is last + 1 (input pointer at p0).")
    w("// Handoff bytes = u32(dims, rows) * u32(dims, cols) * scale; the producer's own u32(last, checkRows/checkCols)")
    w("// must give the same extent.")
    w("struct CutSpec")
    w("{")
    w("    uint8_t cut; // NrPassBorrow::Cut")
    w("    uint16_t first, last, output, dims, rows, cols, checkRows, checkCols;")
    w("    uint32_t scale;")
    w("};")
    w("")
    w("struct Variant")
    w("{")
    w("    const char* model;")
    w("    Runtime runtime;")
    w("    const char* runtimeName;")
    w("    const char* layout;")
    w("    uint16_t clear; // index of cc_cb_clear")
    w("    const Kernel* kernels;")
    w("    uint16_t count;")
    w("    const CutSpec* cuts;")
    w("    uint8_t cutCount;")
    w("};")
    w("")
    cut_ids = {c: i + 1 for i, c in enumerate(CUTS)}
    names = []
    for v in variants:
        ident = f"{v['model']}_{v['runtime'].replace('.', '_')}_{v['layout']}"
        t = v["trace"]
        w(f"// Model {v['model']}, runtime {v['runtime']}, {v['layout']} layout. Traces:")
        for p in v["traces"]:
            w(f"//   {p}")
        w(f"inline constexpr Kernel k{ident}Kernels[] = {{")
        for n, s in zip(t.names, t.sizes):
            w(f'    {{ "{n}", {s} }},')
        w("};")
        for cut, reason in v["unsupported"]:
            w(f"// {cut}: unsupported: {reason}")
        if v["cuts"]:
            w(f"inline constexpr CutSpec k{ident}Cuts[] = {{")
            for c in v["cuts"]:
                w(f"    {{ {cut_ids[c['cut']]}, {c['first']}, {c['last']}, {c['output']}, {c['dims']}, {c['rows']}, "
                  f"{c['cols']}, {c['check_rows']}, {c['check_cols']}, {c['scale']} }}, // {c['cut']}, "
                  f"{c['resolutions']} resolution(s)")
            w("};")
        cuts = f"k{ident}Cuts" if v["cuts"] else "nullptr"
        names.append(f'    {{ "{v["model"]}", Runtime::R{v["runtime"].replace(".", "_")}, "{v["runtime"]}", '
                     f'"{v["layout"]}", {t.clear}, k{ident}Kernels, {len(t)}, {cuts}, {len(v["cuts"])} }},')
        w("")
    w("inline constexpr Variant kVariants[] = {")
    out.extend(names)
    w("};")
    w("")
    w("} // namespace NrPassBorrow::Tables")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as error:
        print(f"error: {error}")
        sys.exit(1)
