"""Compile actual production snippets with small API fixtures; run in an MSVC x64 shell.

--ref accepts a Git revision for checking that the tests reject the old code.
No game or third-party runtime is loaded.
"""
import argparse
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--ref')
args = parser.parse_args()
out = pathlib.Path(tempfile.mkdtemp(prefix='optiscaler-safety-'))


def read(path):
    if args.ref:
        return subprocess.check_output(['git', 'show', f'{args.ref}:{path}'], cwd=ROOT).decode('utf-8-sig')
    return (ROOT / path).read_text(encoding='utf-8-sig')


def function(text, signature):
    start = text.index(signature)
    begin = text.index('{', start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def section(text, first, last):
    start = text.index(first)
    return text[start:text.index(last, start)]


def run(command):
    subprocess.run(command, cwd=out, check=True)


common = r'''
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#define LOG_FUNC(...) ((void)0)
#define LOG_ERROR(...) ((void)0)
#define LOG_WARN(...) ((void)0)
#define LOG_INFO(...) ((void)0)
#define LOG_TRACE(...) ((void)0)
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
'''
util = read('OptiScaler/Util.cpp')
hooks = read('OptiScaler/hooks/Streamline_Hooks.cpp')
version = section(hooks, '        // Keep the Windows path' if '// Keep the Windows path' in hooks
                  else '        char dllPath[MAX_PATH];', '        State::Instance().streamlineVersion.major')
version = version.replace('GetModuleFileNameW(', 'ReadModuleName(')
unicode_test = common + r'''
struct version_t { unsigned major=0, minor=0, patch=0, reserved=0; };
namespace Util { bool GetFileVersion(std::wstring, version_t*, version_t* = nullptr); }
std::wstring string_to_wstring(const std::string& s) {
 int n=MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.c_str(), -1, nullptr, 0);
 if (!n) return {};
 std::wstring result(n, L'\0'); MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.c_str(), -1, result.data(), n);
 result.resize(n-1); return result;
}
int failure=0;
DWORD ReadModuleName(HMODULE m,wchar_t* p,DWORD n) {
 if (failure==1) return 0;
 if (failure==2) { std::fill(p,p+n,L'x'); return n; }
 return GetModuleFileNameW(m,p,n);
}
''' + function(util, 'bool Util::GetFileVersion(') + '\nversion_t Read(HMODULE slInterposer) {\n' + version + r'''
return sl_version;
}
int wmain(int argc,wchar_t** argv) try {
 CHECK(argc==2);
 auto root=std::filesystem::path(L"\\\\?\\"+std::filesystem::current_path().wstring())/L"\u6e38\u620f-\u00e9-\u0416";
 for (int pass=0;pass<2;++pass) {
  if (pass) for(int i=0;i<6;++i) root/=std::wstring(45,L'p');
  std::filesystem::create_directories(root);
  auto path=root/L"fixture.dll";
  std::filesystem::copy_file(argv[1],path,std::filesystem::copy_options::overwrite_existing);
  auto extended=path.wstring();
  HMODULE m=LoadLibraryW(extended.c_str()); CHECK(m);
  auto v=Read(m); CHECK(v.major==21 && v.minor==7 && v.patch==9 && v.reserved==4);
  failure=1; CHECK(Read(m).major==0); failure=2; CHECK(Read(m).major==0); failure=0;
  FreeLibrary(m);
 }
 puts("Unicode and >MAX_PATH module paths, failed/truncated lookup: passed");
 return 0;
} catch (const std::exception& e) { puts(e.what()); return 1; }
'''
(out/'fixture.cpp').write_text('extern "C" __declspec(dllexport) int fixture() { return 1; }')
(out/'fixture.rc').write_text('''#include <windows.h>
1 VERSIONINFO
FILEVERSION 21,7,9,4
PRODUCTVERSION 21,7,9,4
FILETYPE VFT_DLL
BEGIN
 BLOCK "StringFileInfo" BEGIN BLOCK "040904b0" BEGIN VALUE "FileVersion", "21.7.9.4" END END
 BLOCK "VarFileInfo" BEGIN VALUE "Translation", 0x409, 1200 END
END
''')
run(['rc', '/nologo', 'fixture.rc'])
run(['cl', '/nologo', '/LD', 'fixture.cpp', 'fixture.res', '/link', '/OUT:fixture.dll'])

dlssg = read('OptiScaler/framegen/dlssg/DLSSG_Dx12.cpp')
factory = read('OptiScaler/hooks/DxgiFactory_Hooks.cpp')
first = section(dlssg, '    IDXGIFactory* slFactory', '    StreamlineProxy::SetFeatureLoaded()')
second = section(dlssg, '        IDXGIFactory* slFactory', '        IDXGIFactory2* factory2')
third = section(function(factory, 'void DxgiFactoryHooks::HookToDLSSGFactory('), '    IDXGIFactory* real', '    LOG_FUNC();')
com_test = common + r'''
#include <unknwn.h>
using IDXGIFactory = IUnknown;
namespace Util { bool CheckForRealObject(std::string, IUnknown*, IUnknown**); }
IID streamlineRiid {};
struct Fake : IUnknown {
 ULONG refs=1; bool proxy=true;
 HRESULT STDMETHODCALLTYPE QueryInterface(REFIID,void** out) override {
  *out=nullptr; if (!proxy) return E_NOINTERFACE; *out=this; AddRef(); return S_OK;
 }
 ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
 ULONG STDMETHODCALLTYPE Release() override { CHECK(refs>0); return --refs; }
};
int upgrades=0,hooksCalled=0;
namespace StreamlineProxy { auto UpgradeInterface() { return [](void**) { ++upgrades; }; } }
namespace DxgiFactoryHooks { void HookToDLSSGFactory(IUnknown*) { ++hooksCalled; } }
''' + function(util, 'bool Util::CheckForRealObject(') + '\nvoid Legacy(IUnknown* factory) {\n' + first + '\n}\nvoid Hwnd(IUnknown* factory) {\n' + second + '\n}\nvoid Hook(IUnknown* pFactory) {\n' + third + r'''
}
int main() {
 Fake f;
 for(int i=0;i<100;++i) { Legacy(&f); CHECK(f.refs==1); Hwnd(&f); CHECK(f.refs==1); Hook(&f); CHECK(f.refs==1); }
 CHECK(upgrades==0 && hooksCalled==0);
 f.proxy=false; Legacy(&f); Hwnd(&f); Hook(&f); CHECK(f.refs==1); CHECK(upgrades==2 && hooksCalled==2);
 CHECK(f.Release()==0); puts("Three production factory call sites preserve caller ownership: passed");
}
'''
dx = read('OptiScaler/menu/menu_overlay_dx.cpp')
dx_test = common + r'''
#include <d3d12.h>
#include <dxgi1_4.h>
''' + re.search(r'static int const NUM_BACK_BUFFERS = .*?;', dx).group() + r'''
static ID3D12Resource* g_mainRenderTargetResource[NUM_BACK_BUFFERS] {};
static D3D12_CPU_DESCRIPTOR_HANDLE g_mainRenderTargetDescriptor[NUM_BACK_BUFFERS] {};
''' + function(dx, 'static int GetCorrectDXGIFormat(') + '\n' + function(dx, 'static void CreateRenderTargetDx12(') + r'''
int main() {
 CHECK(NUM_BACK_BUFFERS>=16);
 IDXGIFactory4* factory=nullptr; CHECK(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))));
 IDXGIAdapter* warp=nullptr; CHECK(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))));
 ID3D12Device* device=nullptr; CHECK(SUCCEEDED(D3D12CreateDevice(warp,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device))));
 ID3D12CommandQueue* queue=nullptr; D3D12_COMMAND_QUEUE_DESC q{};
 CHECK(SUCCEEDED(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue))));
 ID3D12DescriptorHeap* heap=nullptr; D3D12_DESCRIPTOR_HEAP_DESC h{};
 h.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV; h.NumDescriptors=NUM_BACK_BUFFERS;
 CHECK(SUCCEEDED(device->CreateDescriptorHeap(&h,IID_PPV_ARGS(&heap))));
 auto handle=heap->GetCPUDescriptorHandleForHeapStart(); auto stride=device->GetDescriptorHandleIncrementSize(h.Type);
 for(int i=0;i<NUM_BACK_BUFFERS;++i) { g_mainRenderTargetDescriptor[i]=handle; handle.ptr+=stride; }
 HWND window=CreateWindowExW(0,L"STATIC",L"OptiScaler test",WS_OVERLAPPEDWINDOW,0,0,64,64,nullptr,nullptr,nullptr,nullptr); CHECK(window);
 for(UINT count : {2u,9u,16u,3u}) {
  DXGI_SWAP_CHAIN_DESC1 desc{}; desc.Width=64; desc.Height=64; desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
  desc.SampleDesc.Count=1; desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT; desc.BufferCount=count; desc.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
  IDXGISwapChain1* swap=nullptr; CHECK(SUCCEEDED(factory->CreateSwapChainForHwnd(queue,window,&desc,nullptr,nullptr,&swap)));
  CreateRenderTargetDx12(device,swap);
  for(UINT i=0;i<count;++i) { CHECK(g_mainRenderTargetResource[i]); g_mainRenderTargetResource[i]->Release(); g_mainRenderTargetResource[i]=nullptr; }
  swap->Release();
 }
 DestroyWindow(window); heap->Release(); queue->Release(); device->Release(); warp->Release(); factory->Release();
 puts("Production DX12 target creation on WARP: 2, 9, 16, then 3 buffers passed");
}
'''
# Execute the actual render-index gate with missing resources and out-of-range indices.
guard = section(dx, '                UINT backBufferIdx =', '                ID3D12CommandAllocator* commandAllocator')
guard_code = r'''
struct IndexFixture {
 UINT index; unsigned releases=0;
 UINT GetCurrentBackBufferIndex() { return index; }
 void Release() { ++releases; }
};
ID3D12CommandAllocator* g_commandAllocators[NUM_BACK_BUFFERS] {};
unsigned rendered=0;
void CheckIndex(IndexFixture* pSwapChain) {
''' + guard + r'''
 ++rendered;
}
void CheckRenderGuards() {
 IndexFixture fixture{16}; CheckIndex(&fixture); CHECK(rendered==0 && fixture.releases==1);
 fixture.index=15; CheckIndex(&fixture); CHECK(rendered==0 && fixture.releases==2);
 g_mainRenderTargetResource[15]=reinterpret_cast<ID3D12Resource*>(1);
 CheckIndex(&fixture); CHECK(rendered==0 && fixture.releases==3);
 g_commandAllocators[15]=reinterpret_cast<ID3D12CommandAllocator*>(1);
 CheckIndex(&fixture); CHECK(rendered==1 && fixture.releases==3);
 g_mainRenderTargetResource[15]=nullptr; g_commandAllocators[15]=nullptr;
}
'''
if 'backBufferIdx >= NUM_BACK_BUFFERS' in guard:
    dx_test = dx_test.replace('int main() {', guard_code + '\nint main() {\n CheckRenderGuards();')
vk = read('OptiScaler/menu/menu_overlay_vk.cpp')
vk_test = common + r'''
using VkDevice=int; using VkSwapchainKHR=int; using VkImage=unsigned long long; using VkResult=int;
constexpr int VK_SUCCESS=0,VK_INCOMPLETE=5;
unsigned wanted=16,queries=0,fills=0; int scenario=0;
int vkGetSwapchainImagesKHR(int,int,unsigned* n,VkImage* p) {
 if(!p) { ++queries; *n=wanted; return scenario==1 ? -1 : VK_SUCCESS; }
 ++fills; if(scenario==2) return -1;
 if(scenario==3 && fills==1) { wanted=20; return VK_INCOMPLETE; }
 if(scenario==4) return VK_INCOMPLETE;
 if(scenario==5) *n=3;
 for(unsigned i=0;i<*n;++i) p[i]=100+i;
 return VK_SUCCESS;
}
'''
if 'static bool GetMenuSwapchainImages(' in vk:
    vk_test += function(vk, 'static bool GetMenuSwapchainImages(')
else:
    vk_test = None
if vk_test is not None:
 vk_test += r'''
int main() {
 std::vector<VkImage> images;
 for(unsigned n : {2u,9u,16u,24u,3u}) { wanted=n; CHECK(GetMenuSwapchainImages(1,1,images)); CHECK(images.size()==n && images.back()==99+n); }
 wanted=1; CHECK(!GetMenuSwapchainImages(1,1,images)); wanted=0; CHECK(!GetMenuSwapchainImages(1,1,images));
 wanted=16; scenario=1; CHECK(!GetMenuSwapchainImages(1,1,images)); scenario=2; CHECK(!GetMenuSwapchainImages(1,1,images));
 scenario=3; fills=0; CHECK(GetMenuSwapchainImages(1,1,images)); CHECK(images.size()==20 && fills==2);
 scenario=4; fills=0; CHECK(!GetMenuSwapchainImages(1,1,images)); CHECK(fills==3 && images.empty());
 scenario=5; CHECK(GetMenuSwapchainImages(1,1,images)); CHECK(images.size()==3);
 puts("Vulkan large/changing counts, shrink, errors, and bounded VK_INCOMPLETE retries: passed");
}
'''
failures = []
for name, code, libs in [('unicode', unicode_test, ['version.lib']), ('factory', com_test, ['ole32.lib']),
                         ('dx_buffers', dx_test, ['d3d12.lib', 'dxgi.lib', 'user32.lib']), ('vk_buffers', vk_test, [])]:
    if code is None:
        print(name, 'helper absent at this revision; skipped')
        continue
    (out/f'{name}.cpp').write_text(code, encoding='utf-8')
    run(['cl', '/nologo', '/std:c++20', '/EHsc', '/O2', '/utf-8', f'{name}.cpp', '/link', *libs])
    result = subprocess.run([str(out/f'{name}.exe'), str(out/'fixture.dll')], cwd=out)
    if result.returncode:
        failures.append(name)
print('Artifacts:', out)
print('Failed categories:', failures)
raise SystemExit(bool(failures))
