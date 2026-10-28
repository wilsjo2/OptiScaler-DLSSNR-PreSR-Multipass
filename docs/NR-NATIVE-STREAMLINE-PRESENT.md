# Finished-picture NR with native Streamline FG

During native DLSSG, Streamline's display buffers differ from the game's finished buffers. NR edits underneath the interposer can be overwritten by its later copy. See NVIDIA's [interposer](https://github.com/NVIDIA-RTX/Streamline/blob/main/source/core/sl.interposer/dxgi/dxgiSwapchain.cpp) and [hook ABI](https://github.com/NVIDIA-RTX/Streamline/blob/main/source/core/sl.api/internal.h).

NR wraps the native DLSSG plugin's legacy and HWND swapchain creation callbacks, plus Present/Present1. Creation records the original game queue. Before DLSSG presents, NR retrieves the app-facing index/buffer through that plugin's hooks and processes it on the game queue. Declined hooks skip NR without substituting a display buffer. Test presents and disabled NR do no work; OptiScaler's local DLSSG hook keeps its existing KCD2 and real-DLSSG checks.

Native handoff selects matching submitted input by queue order/completion, rather than the inner display counter, which includes generated frames. Other routes retain their epoch check. Unfinished cross-queue input cannot add a presentation wait: that previously caused a Cyberpunk freeze.

`tests/nr_streamline_picture_smoke.cpp` rotates app buffers on WARP, demonstrates overwritten display edits versus surviving app edits, and rejects declined hooks. `tests/nr_streamline_hooks.cpp` checks queue registration and application-buffer selection through both creation callbacks, including legacy creation by the game's plugin. These tests do not load NVIDIA's model or establish interpolation quality; see [game observations](NR-UPSTREAM-REVIEW.md).
