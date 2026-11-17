#include "Processor.h"
#include <DirectXPackedVector.h>
#include <d3d12sdklayers.h>
#include <cmath>
#include <iostream>
#include <vector>
// Re-apply and pass-plan checks without NGX: the CPU pass plan, then shader mode 22 on WARP.
namespace nr {
namespace {
using Pixel = std::array<float, 4>;
void Require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void PlanCheck() {
    using NrPassBorrow::Cut;
    std::array<Cut, Settings::MaxPasses> cuts{}; unsigned evaluations = 0;
    Settings s; s.unlockPasses = true; s.passes = 5;
    Require(!BorrowCuts(s, cuts, evaluations) && evaluations == 5 && s.ReapplyMask() == 0, "Default passes borrowed or re-applied.");
    s.passShortcuts[0].reapply = true; s.passShortcuts[0].borrowBelow = true; // Re-apply wins over borrowing.
    s.passShortcuts[1].borrowBelow = true; s.passShortcuts[1].borrowTransformer = true; s.passShortcuts[1].borrowLevel = 16;
    s.passShortcuts[2].borrowTransformer = true;
    s.passShortcuts[3].reapply = true;
    s.passShortcuts[5].borrowTransformer = true; s.passShortcuts[5].reapply = true; // Inactive pass 7: ignored.
    Require(BorrowCuts(s, cuts, evaluations), "Borrowing passes were not reported.");
    Require(evaluations == 3 && cuts[0] == Cut::None && cuts[1] == NrPassBorrow::ChooseCut(true, true, 16) &&
        cuts[2] == NrPassBorrow::ChooseCut(true, false, 32) && cuts[3] == Cut::None, "Cuts are not listed per network evaluation.");
    Require(s.ReapplyMask() == ((1u << 1) | (1u << 4)), "Re-apply mask included pass 1 or an inactive pass.");
    Require(NrPassBorrow::ChooseCut(true, true, 16) == Cut::Below16 && NrPassBorrow::ChooseCut(true, false, 32) == Cut::Transformer &&
        NrPassBorrow::ChooseCut(false, false, 8) == Cut::None, "ChooseCut does not follow the tick-boxes (below wins).");
    s.passes = 1;
    Require(!BorrowCuts(s, cuts, evaluations) && evaluations == 1 && s.ReapplyMask() == 0, "A single pass borrowed or re-applied.");
    std::cout << "PASS pass plan: re-apply evaluates nothing and wins over borrowing; cuts per network evaluation; inactive passes ignored\n";
}
}
struct ReapplyProbe {
    static void Shader() {
        using namespace DirectX::PackedVector;
        Gpu g(true); Processor p(g);
        constexpr UINT w = 64, h = 4;
        const auto readable = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        auto after = g.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, readable), before = g.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, readable);
        auto result = g.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, readable);
        const float nan = std::nanf(""), inf = INFINITY;
        // Row 0: ordinary encoded values; row 1: out-of-range network answers; rows 2-3: non-finite out or in.
        auto outValue = [&](UINT x, UINT y) -> Pixel {
            const float t = (x + .5f) / w;
            if (y == 1) return {t * 1.6f - .3f, 1.2f - t, .5f, .5f};
            if (y == 2) return {x % 3 == 0 ? nan : t, x % 3 == 1 ? inf : t, x % 3 == 2 ? -inf : t, .75f};
            return {t, 1 - t, .25f + .5f * t, .25f};
        };
        auto inValue = [&](UINT x, UINT y) -> Pixel {
            const float t = (x + .5f) / w;
            if (y == 3) return {x % 2 ? nan : .4f, x % 2 ? .4f : inf, .4f, 1};
            return {.5f, .5f * t, .6f - .2f * t, 1};
        };
        auto halfRound = [](float v) { return XMConvertHalfToFloat(XMConvertFloatToHalf(v)); };
        auto upload = [&](ID3D12Resource* texture, auto&& value) {
            const auto d = texture->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{}; UINT64 bytes = 0;
            g.device->GetCopyableFootprints(&d, 0, 1, 0, &layout, nullptr, nullptr, &bytes);
            auto buffer = g.Buffer(bytes, D3D12_HEAP_TYPE_UPLOAD); void* mapped = nullptr; Check(buffer->Map(0, nullptr, &mapped), "Re-apply upload");
            for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) {
                auto* row = reinterpret_cast<HALF*>(static_cast<BYTE*>(mapped) + layout.Offset + size_t(y) * layout.Footprint.RowPitch);
                const auto v = value(x, y); for (UINT c = 0; c < 4; ++c) row[x * 4 + c] = XMConvertFloatToHalf(v[c]);
            }
            buffer->Unmap(0, nullptr); g.Begin();
            Barrier(g.commands.Get(), texture, readable, D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION src{}, dst{}; src.pResource = buffer.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = layout; dst.pResource = texture; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            g.commands->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            Barrier(g.commands.Get(), texture, D3D12_RESOURCE_STATE_COPY_DEST, readable);
            g.Submit(); if (!g.Drain(30000)) throw std::runtime_error("Re-apply upload timeout.");
        };
        upload(after.Get(), outValue); upload(before.Get(), inValue);
        const auto sanitize = [](float v, float fallback) { return std::isfinite(v) ? v : fallback; };
        for (float gain : {0.f, .81f, 1.f, 2 * .81f, 2.f}) {
            DlssNrConstants c{}; c.Mode = 22; c.Width = w; c.Height = h; c.TransferStrength = gain;
            g.Begin(); Barrier(g.commands.Get(), result.Get(), readable, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            p.Dispatch(Processor::ReapplySlot(1), c, after.Get(), before.Get(), nullptr, result.Get(), nullptr);
            Barrier(g.commands.Get(), result.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            const auto d = result->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{}; UINT64 bytes = 0;
            g.device->GetCopyableFootprints(&d, 0, 1, 0, &layout, nullptr, nullptr, &bytes);
            auto readback = g.Buffer(bytes, D3D12_HEAP_TYPE_READBACK);
            D3D12_TEXTURE_COPY_LOCATION src{}, dst{}; src.pResource = result.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.pResource = readback.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = layout;
            g.commands->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            Barrier(g.commands.Get(), result.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, readable);
            g.Submit(); if (!g.Drain(30000)) throw std::runtime_error("Re-apply shader timeout.");
            Diagnostics(g);
            void* mapped = nullptr; Check(readback->Map(0, nullptr, &mapped), "Re-apply readback");
            float worst = 0;
            for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) {
                const auto* row = reinterpret_cast<const HALF*>(static_cast<const BYTE*>(mapped) + layout.Offset + size_t(y) * layout.Footprint.RowPitch);
                auto o = outValue(x, y), i = inValue(x, y);
                for (auto& v : o) v = std::isfinite(v) ? halfRound(v) : v;
                for (auto& v : i) v = std::isfinite(v) ? halfRound(v) : v;
                for (UINT ch = 0; ch < 3; ++ch) {
                    const float a = sanitize(o[ch], .5f), b = sanitize(i[ch], a);
                    const float expected = std::clamp(a + gain * (a - b), 0.f, 1.f), actual = XMConvertHalfToFloat(row[x * 4 + ch]);
                    Require(std::isfinite(actual), "Re-apply returned a non-finite value.");
                    worst = std::max(worst, std::abs(actual - expected));
                }
                Require(XMConvertHalfToFloat(row[x * 4 + 3]) == o[3], "Re-apply did not keep the network answer's alpha.");
            }
            readback->Unmap(0, nullptr);
            if (worst > 1e-3f) throw std::runtime_error(std::format("Re-apply gain {} differs from saturate(out + g(out - in)) by {}.", gain, worst));
            std::cout << std::format("PASS re-apply shader gain {:.2f}: max error {:.6f} (saturation, non-finite inputs, alpha)\n", gain, worst);
        }
    }
    static void Diagnostics(Gpu& gpu) {
        ComPtr<ID3D12InfoQueue> messages; gpu.device.As(&messages);
        if (messages) for (UINT64 i = 0; i < messages->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
            SIZE_T bytes = 0; messages->GetMessage(i, nullptr, &bytes); std::vector<BYTE> storage(bytes);
            auto* m = reinterpret_cast<D3D12_MESSAGE*>(storage.data()); messages->GetMessage(i, m, &bytes);
            if (m->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) throw std::runtime_error(m->pDescription);
        }
    }
};
void ReapplyCheck() { PlanCheck(); ReapplyProbe::Shader(); }
}
