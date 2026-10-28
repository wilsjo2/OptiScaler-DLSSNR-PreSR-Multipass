# v0.8.91 prerelease - compressed model input preview

Adds **Debug view → Compressed model input** and a **Preview** checkbox directly below **Peripheral compression**. Both control the same saved setting (`DebugView=4`) on DX12 and native Vulkan, including the existing DX12 bridges.

The preview displays the immutable input passed to NR before spatial unpacking, scaled to fill the screen. This makes the squeezed peripheral geometry visible. Without active compression it shows ordinary model input. Apply model must be enabled. Existing Proxy and Model output debug views retain unpacked geometry; unticking Preview turns debug view off.

Retains v0.8.9's spatial compression and compatibility behavior. Standard and optional RTX40-MFG packages are provided; preserve game-specific INI settings when updating. NVIDIA model/FG runtime DLLs are not bundled. Live game preview and RTX40 hardware validation have not been performed.
