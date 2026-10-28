#pragma once
#include "DlssNr_Readback.h"
#include "DlssNr_GpuLifetime.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace capture
{
constexpr unsigned int kMaxFrames = 8;

// Consecutive before/after images, with completion tracked for every recorded copy.
class FrameCapture
{
    struct Pair
    {
        DlssNr::ReadbackImage before, after;
    };
    struct Data
    {
        std::vector<Pair> pairs;
        Microsoft::WRL::ComPtr<ID3D12QueryHeap> query;
        Microsoft::WRL::ComPtr<ID3D12Resource> stamps;
        unsigned recorded = 0;

        bool Init(ID3D12Device* device, ID3D12Resource* before, ID3D12Resource* after, unsigned count)
        {
            pairs.resize(count);
            for (auto& pair : pairs)
                if (!pair.before.Allocate(device, before->GetDesc()) || !pair.after.Allocate(device, after->GetDesc()))
                    return false;
            D3D12_QUERY_HEAP_DESC desc { D3D12_QUERY_HEAP_TYPE_TIMESTAMP, count, 0 };
            if (FAILED(device->CreateQueryHeap(&desc, IID_PPV_ARGS(&query))) ||
                !DlssNr::CreateReadbackBuffer(device, count * sizeof(UINT64), &stamps))
                return false;
            void* mapped = nullptr;
            D3D12_RANGE empty {}, written { 0, count * sizeof(UINT64) };
            if (FAILED(stamps->Map(0, &empty, &mapped)))
                return false;
            std::memset(mapped, 0, written.End);
            stamps->Unmap(0, &written);
            return true;
        }

        bool AllSubmitted() const
        {
            void* mapped = nullptr;
            D3D12_RANGE range { 0, recorded * sizeof(UINT64) }, empty {};
            if (FAILED(stamps->Map(0, &range, &mapped)))
                return false;
            const auto* values = static_cast<const UINT64*>(mapped);
            const bool complete = std::all_of(values, values + recorded, [](UINT64 value) { return value != 0; });
            stamps->Unmap(0, &empty);
            return complete;
        }
    };
    std::unique_ptr<Data> data_;
    DlssNr::GpuLifetime lifetime_;
    unsigned wanted_ = 0;

    static bool Matches(const DlssNr::ReadbackImage& image, ID3D12Resource* source)
    {
        const auto desc = source->GetDesc();
        const auto& stored = image.layout.Footprint;
        return stored.Width == desc.Width && stored.Height == desc.Height &&
               stored.Format == DlssNr::TypedReadbackFormat(desc.Format);
    }

  public:
    ~FrameCapture() { release(); }
    void request(unsigned frames)
    {
        if (!isActive())
            wanted_ = std::min(frames, kMaxFrames);
    }
    bool isActive() const { return wanted_ != 0; }
    void Submitted(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists)
    {
        lifetime_.Submitted(queue, count, lists);
    }
    void ResetRecording(ID3D12CommandList* commands) { lifetime_.ResetRecording(commands); }
    void FinishSubmitted() { lifetime_.FinishSubmitted(); }

    void record(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, ID3D12Resource* before,
                D3D12_RESOURCE_STATES beforeState, ID3D12Resource* after, D3D12_RESOURCE_STATES afterState)
    {
        if (!isActive() || (data_ && data_->recorded == wanted_))
            return;
        if (!data_)
        {
            auto next = std::make_unique<Data>();
            if (!next->Init(device, before, after, wanted_))
            {
                release();
                return;
            }
            data_ = std::move(next);
        }
        auto& pair = data_->pairs[data_->recorded];
        if (!Matches(pair.before, before) || !Matches(pair.after, after))
        {
            const auto retry = wanted_;
            release();
            request(retry);
            return;
        }
        lifetime_.Record(cmd);
        pair.before.Copy(cmd, before, beforeState);
        pair.after.Copy(cmd, after, afterState);
        cmd->EndQuery(data_->query.Get(), D3D12_QUERY_TYPE_TIMESTAMP, data_->recorded);
        cmd->ResolveQueryData(data_->query.Get(), D3D12_QUERY_TYPE_TIMESTAMP, data_->recorded, 1, data_->stamps.Get(),
                              data_->recorded * sizeof(UINT64));
        ++data_->recorded;
    }

    std::string write(const std::filesystem::path& directory)
    {
        if (!data_ || data_->recorded != wanted_ || !lifetime_.Idle())
            return {};
        if (!data_->AllSubmitted())
        {
            const auto retry = wanted_;
            release();
            request(retry);
            return {};
        }
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        bool success = !error;
        for (unsigned i = 0; success && i < wanted_; ++i)
        {
            char name[32];
            std::snprintf(name, sizeof(name), "before_%02u.raw", i);
            success = data_->pairs[i].before.Write(directory / name);
            std::snprintf(name, sizeof(name), "after_%02u.raw", i);
            success &= data_->pairs[i].after.Write(directory / name);
        }
        if (success)
        {
            std::ofstream manifest(directory / "manifest.txt");
            manifest << "frames " << wanted_ << '\n';
            const auto describe = [&](const char* name, const DlssNr::ReadbackImage& image)
            {
                const auto& f = image.layout.Footprint;
                manifest << name << " width " << f.Width << " height " << f.Height << " format " << int(f.Format)
                         << " rowPitch " << f.RowPitch << '\n';
            };
            describe("before", data_->pairs.front().before);
            describe("after", data_->pairs.front().after);
            manifest << "\nbefore_NN.raw is the frame as the upscaler produced it.\n"
                        "after_NN.raw is the same frame once the model's edit was applied.\n"
                        "Consecutive frames, same run, so the pair is a control.\n";
            manifest.close();
            success = !manifest.fail();
        }
        release();
        return success ? directory.string() : std::string {};
    }

    void release()
    {
        if (auto* retired = data_.release())
            lifetime_.Retire([retired] { delete retired; });
        lifetime_.BeginGeneration();
        wanted_ = 0;
    }
};
} // namespace capture
