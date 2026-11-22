#pragma once
#include <filesystem>
#include <unknwn.h>
// The shared retirement utility also serves wrapped game queues. This executable
// only creates native COM objects: there are no wrapper identities to resolve.
namespace Util {
inline bool CheckForRealObject(const char*, IUnknown*, IUnknown**) { return false; }
// No OptiScaler.dll here: the model wrapper's INT8 path (which looks for its files beside that DLL) is not used.
inline std::filesystem::path DllPath() { return {}; }
}
