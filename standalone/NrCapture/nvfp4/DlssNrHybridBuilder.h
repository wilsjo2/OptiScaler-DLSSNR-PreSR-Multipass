#pragma once
// Descriptor encoding only: no CUDA kernel launch, allocation, or renderer copies.
struct FusedInput{int m,n,k,sm_count;uint64_t a,b,sfa,sfb,c,d,sfd,norm,workspace;};
struct FusedInfo{uint64_t params_bytes,workspace_bytes,shared_bytes,sfa_bytes,sfb_bytes,sfd_bytes;uint32_t grid[3],block[3];};
static_assert(sizeof(FusedInput)==88&&sizeof(FusedInfo)==72,"Fused helper ABI");
struct FusedBuilder{
 HMODULE helper=nullptr,driver=nullptr;void*ctx=nullptr;
 int(__cdecl*build)(const FusedInput*,void*,uint64_t,FusedInfo*)=nullptr;
 int(__stdcall*push)(void*)=nullptr;int(__stdcall*pop)(void**)=nullptr;
 template<class T>T sym(HMODULE m,const char*n){auto f=GetProcAddress(m,n);if(!f)throw std::runtime_error(std::string("Missing fused helper/driver export ")+n);return reinterpret_cast<T>(f);}
 void init(const std::filesystem::path&dir){if(helper)return;
  helper=LoadLibraryExW((dir/L"params_builder.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS|LOAD_LIBRARY_SEARCH_USER_DIRS);
  if(!helper)throw std::runtime_error("Cannot load fused params_builder.dll: "+std::to_string(GetLastError()));
  build=sym<decltype(build)>(helper,"fused_params");driver=LoadLibraryW(L"nvcuda.dll");if(!driver)throw std::runtime_error("CUDA driver unavailable for descriptor encoding");
  auto init=sym<int(__stdcall*)(unsigned)>(driver,"cuInit");auto deviceGet=sym<int(__stdcall*)(int*,int)>(driver,"cuDeviceGet");auto retain=sym<int(__stdcall*)(void**,int)>(driver,"cuDevicePrimaryCtxRetain");
  push=sym<decltype(push)>(driver,"cuCtxPushCurrent_v2");pop=sym<decltype(pop)>(driver,"cuCtxPopCurrent_v2");int device=0;
  if(init(0)||deviceGet(&device,0)||retain(&ctx,device))throw std::runtime_error("CUDA primary context unavailable for CPU TMA encoding");
 }
 void encode(const FusedInput&input,std::vector<unsigned char>&params,FusedInfo&info){
  int rc=build(&input,nullptr,0,&info);if(rc||info.params_bytes!=2176||info.workspace_bytes)throw std::runtime_error("Unsupported fused Params/workspace ABI status "+std::to_string(rc));params.resize(info.params_bytes);
  if(push(ctx))throw std::runtime_error("CUDA context push for TMA encoding failed");
  rc=build(&input,params.data(),params.size(),&info);void*previous=nullptr;int popped=pop(&previous);
  if(rc||popped)throw std::runtime_error("Fused D3D-address encoding failed status "+std::to_string(rc)+" pop "+std::to_string(popped));
  if(!info.grid[0]||!info.grid[1]||!info.grid[2]||!info.block[0]||info.workspace_bytes)throw std::runtime_error("Invalid fused launch geometry");
 }
};
