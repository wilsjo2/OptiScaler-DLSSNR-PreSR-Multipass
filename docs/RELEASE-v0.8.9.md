# v0.8.9 prerelease - peripheral spatial compression

Adds optional peripheral compression to DX12 and native Vulkan NR, including the existing DX12 bridges. Enable it under Model resolution; the default 80% centre / 90% working layout processes 81% of the pixels at 100% Model resolution. Independent horizontal/vertical sizes, offsets and shifts combine with Model resolution, including supersampling. Lowering Model resolution also lowers centre detail. Compression is disabled by default, and existing configurations retain their settings.

Colour, depth and endpoint-transformed motion use the same packed geometry. The immutable model input and final answer are unpacked together before existing reconstruction. Layout changes reset histories. Spatial failures preserve the current frame and fall back to ordinary NR. Optional outlines are drawn after NR. See [controls, integration and attribution](NR-SPATIAL-COMPRESSION.md).

This release retains v0.8.8 reconstruction modes and includes the RDR2 finished-picture queue candidate: composition is ordered on the XeFG game queue before presentation. It does not claim to fix the separate RDR2 HDR highlight flicker report.

The full NR prerelease suite passed, including the RDR2 queue regression, CPU mapping tests and production Vulkan shader fixtures. Standard and RTX40-MFG builds are provided. On an RTX 5090, two synthetic actual-NR 4K benchmark runs measured 10.6–12.3% lower pack/model/unpack time with the default layout and 17.8–18.9% lower at combined 85% Model resolution. These measurements exclude host colour encoding, final composition and game rendering; they are not whole-game FPS predictions.

Native Vulkan's actual NGX evaluation with packed RGBA32F motion still needs game validation. Synthetic tests do not establish live game quality, HDR, reconstruction combinations, frame hold, resize/layout changes, failure injection or shutdown behavior. RTX40 MFG has not been verified on RTX40 hardware. No game was launched for validation.

Use the standard ZIP unless you need the optional RTX40-MFG build. Preserve existing game-specific INI settings when updating. NVIDIA model/FG runtime DLLs are not bundled. The MIT notice for BeliyG3's pinned spatial algorithm is included in both packages. Previous releases remain available.
