#pragma once
#include <nvsdk_ngx_defs.h>
#include <d3d12.h>

namespace DlssNr::NgxDiagnostics
{
// Install before driver initialization; preserve the caller's original logging sink.
void Install(NVSDK_NGX_LoggingInfo& logging);
class Scope
{
  public:
    Scope();
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
void RuntimeReport(ID3D12GraphicsCommandList* commands, ID3D12Device* device, const char* phase);
} // namespace DlssNr::NgxDiagnostics
