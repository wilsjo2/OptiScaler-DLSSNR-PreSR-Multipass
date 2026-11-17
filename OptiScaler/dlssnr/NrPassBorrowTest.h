#pragma once

// Test seam for NrPassBorrow.cpp, used by OptiScaler/tests/nr_pass_borrow_smoke.cpp. Production code never calls it.
// The tests obtain the NvAPI wrappers through NrPassBorrow::WrapInterface with fake originals and call them directly.

#include "NrPassBorrow.h"

namespace NrPassBorrow::Test
{

// One handoff copy the module wants recorded on the command list (UAV barrier, copy, UAV barrier).
struct Copy
{
    bool toSnapshot;     // true: arena -> this module's snapshot of `cut`; false: snapshot of `cut` -> arena
    Cut cut;
    uint64_t arena;      // GPU virtual address in the NR runtime's pool: producer output or consumer input
    uint64_t bytes;
    unsigned evaluation; // evaluation of the frame that requested it
};
using CopyFn = bool (*)(ID3D12GraphicsCommandList* list, const Copy& copy, void* user); // false: the copy failed
// Replace the D3D12 copy (snapshot allocation, barriers and dispatch). nullptr restores the D3D12 path.
void SetCopy(CopyFn fn, void* user);

enum class Gate : uint8_t
{
    Checking,
    NotAudited,
    R310_8,
    R310_9_16,
};
using GateFn = Gate (*)(void* user);
// Replace the SHA-256 runtime allowlist for every module. nullptr restores the hash gate.
void SetRuntimeGate(GateFn fn, void* user);

} // namespace NrPassBorrow::Test
