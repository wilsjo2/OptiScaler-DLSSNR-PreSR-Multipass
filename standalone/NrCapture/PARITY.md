# Standalone NR feature/control parity

This build implements the in-game NR controls that apply to a finished captured picture. It does not claim equivalent live image quality or engine access.

| Area | Standalone support |
| --- | --- |
| Model passes | 1–2 normally, unlock up to 10; independent feature handles, tuning and history |
| Pass editing | All ten profiles can be edited while inactive; later passes inherit pass 1 with tone 0, or use a complete custom profile |
| Model controls | Intensity, local structure, local tone, skin structure, Standard/Natural/Cinematic style, Automatic/Original/Optimized model labels (runtime support required), legacy preset 3, auto skin mask |
| Working resolution | 25–200%; captured output dimensions stay fixed |
| Supersampling | Same compiled Bicubic input enlargement and eight reduction filters as OptiScaler; FSR1 selection uses its own shader for both directions |
| Subnative enlargement | Classic, matched residuals and private DLSS SR, plus spatial edge-aware enlargement; synthetic guides remain a limitation |
| Halo suppression | Shared dark-rim filters around bright/dark objects, strengths 0–1, radius 1–4 NR pixels; disabled by default |
| Colour | All five HDR mappings; reference white; explicit SDR/HDR input interpretation |
| Composition | Detail and colour strengths, 1–8× highlight guard, independent skin/environment detail and colour, mask preview |
| Inspection | Frozen capture, hidden model edit while NR runs, original view, wipe/split, side-by-side/zoom, swap, labels/size, three debug views |
| Standalone operation | Monitor/window selection, bounded capture queue, optional motion estimation, adjustable FPS limit, stop hotkey |

Pass inheritance is selected for the whole profile instead of separate per-field reset buttons. The labels use a small built-in bitmap font rather than the game's ImGui font. Hold/inspection states are session-only so the application starts with an ordinary live processed picture. Existing INI values are retained; new profile and composition fields are stored alongside them.

Game-owned features have no external-capture equivalent: pre-SR placement, carrying an early edit across SR/RR, fitting that edit to a later engine HDR buffer, native depth/motion/exposure guides, game frame generation, graphics-API bridges and game quirks. The standalone does not add hooks or access another process to provide these.

## Verification

`NrCaptureCheck --feature-check 1` uses saved/synthetic pixels, never a game or visible window. With the D3D12 debug layer enabled it verifies:

- Exact settings round trip using an isolated temporary INI.
- 1 → 10 → 1 → 10 pass transitions, a visible pass-10 override, and inactive-profile isolation.
- Style/mask tuning changes, frozen-frame identity across source changes/model resizing, hidden edit, and live resume.
- All eight supersampling filters, 200% model size, and an exact return to native-size output.
- Highlight guard, three debug outputs, exact original/processed wipe regions, side-by-side zoom and label scaling.

The existing appearance/motion/private-DLSS checks and matched-input OptiScaler model-wrapper comparison are also retained. GUI appearance and live gameplay remain for user verification. See `VALIDATION.md` for run results.
