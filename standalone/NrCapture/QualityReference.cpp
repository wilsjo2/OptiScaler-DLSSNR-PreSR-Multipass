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
// The reference uses NVIDIA models. Custom pass checks use Processor and its native host.
namespace DlssNr::Nvfp4 {
std::shared_ptr<void> CreateHost(ID3D12Device*, unsigned, unsigned, unsigned, const std::filesystem::path&) { return nullptr; }
bool RunHost(const std::shared_ptr<void>&, ID3D12GraphicsCommandList*, const HostPass&) { return false; }
}
