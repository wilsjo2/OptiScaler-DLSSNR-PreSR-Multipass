// Compile OptiScaler's real model wrapper into the headless check only.
#include "../../OptiScaler/dlssnr/DlssNr_Proxy.cpp"

namespace DlssNr::NgxDiagnostics {
Scope::Scope() {}
Scope::~Scope() {}
void RuntimeReport(ID3D12GraphicsCommandList*, ID3D12Device*, const char*) {}
}
namespace DlssNr {
std::shared_ptr<CompatibilityRuntime> CompatibilityRuntime::TryOpen(ID3D12Device*) { return nullptr; }
}
// The wrapper's INT8 path is OptiScaler's; this program runs the INT8 models through Processor (nvfp4/NrHost.h).
namespace DlssNr::Nvfp4 {
std::shared_ptr<void> CreateInt8(ID3D12Device*, unsigned, unsigned, unsigned, const std::filesystem::path&) { return nullptr; }
bool RunInt8(const std::shared_ptr<void>&, ID3D12GraphicsCommandList*, const Int8Pass&) { return false; }
}
