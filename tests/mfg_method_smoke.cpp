// Host check of MfgUnlockMethod.h: which temporal fix the configuration selects.
// cl /std:c++20 /EHsc tests/mfg_method_smoke.cpp
#include "../OptiScaler/framegen/dlssg/MfgUnlockMethod.h"

#include <cstdio>

using namespace MfgUnlock;

static int fails = 0;
#define CHECK(c)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(c))                                                                                                      \
        {                                                                                                              \
            printf("FAIL line %d: %s\n", __LINE__, #c);                                                                \
            ++fails;                                                                                                   \
        }                                                                                                              \
    } while (0)

using Fix = std::optional<std::string>;

int main()
{
    // An ini without the key, or one saved with the default: the method the unlock shipped with.
    CHECK(ResolveTemporalMethod(std::nullopt) == TemporalMethod::Retarget);
    CHECK(ResolveTemporalMethod(Fix("Auto")) == TemporalMethod::Retarget);

    // A named method is taken as named.
    CHECK(ResolveTemporalMethod(Fix("Retarget")) == TemporalMethod::Retarget);
    CHECK(ResolveTemporalMethod(Fix("Ptx")) == TemporalMethod::Ptx);

    // Anything else is Auto. Config normalises the case before it gets here.
    CHECK(ResolveTemporalMethod(Fix("")) == TemporalMethod::Retarget);
    CHECK(ResolveTemporalMethod(Fix("ptx")) == TemporalMethod::Retarget);
    CHECK(ResolveTemporalMethod(Fix("Off")) == TemporalMethod::Retarget);

    if (fails == 0)
        printf("mfg_method_smoke: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
