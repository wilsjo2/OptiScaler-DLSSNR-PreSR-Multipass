# Credits

This fork builds on [Dagherbou/OptiScaler_DLSSNR](https://github.com/Dagherbou/OptiScaler_DLSSNR) and [OptiScaler](https://github.com/optiscaler/OptiScaler). OptiScaler began with [PotatoOfDoom's CyberFSR2](https://github.com/PotatoOfDoom/CyberFSR2).

Colour processing derives from [clshortfuse's RenoDX](https://github.com/clshortfuse/renodx); see [attribution/licence](../Licenses/RenoDX_ATTRIBUTION.txt).

Integrations include hhkbble's multipass/composition, [y4my4my4m's Vulkan work](NR-VULKAN.md) and [cmh1448's motion metadata](NR-MOTION-METADATA.md). Linked notes identify source commits and test limits.

This variant restores the built-in [Ada MFG work and attribution](RTX40-MFG.md), derived from y4my4my4m and the earlier fork under GPL-3.0.

## RTX 40 MFG unlock

The built-in RTX 40 multi frame generation unlock is adapted from [y4my4my4m's fork](https://github.com/y4my4my4m/OptiScaler_DLSSNR_Multipass_MFG) (GPL-3.0). The provider discovery, the Streamline plugin frame-ceiling patch, the software frame pacing option and the PTX temporal fix are adapted from [KleberMotta's fork](https://github.com/KleberMotta/OptiScaler-DLSS5-MFG-RTX40) (MIT), a port of the MFG Unlock ReShade addon by [Dreamt](https://github.com/ImDreamt/MFGAdaUnlock-RenoDx) and [mavismmg](https://github.com/mavismmg/MFGAdaUnlock-RenoDx). The technique originates from [dashdogy's RTX40MFG-Unlock](https://github.com/dashdogy/RTX40MFG-Unlock). See [RTX40-MFG.md](RTX40-MFG.md) and the [licences](../Licenses/MFGUnlock_LICENSE.txt).

## OptiScaler contributors

- @PotatoOfDoom for CyberFSR2.
- @Artur for DLSS Enabler and help with the NVNGX API.
- @LukeFZ and @Nukem for their mods and shared knowledge.
- @FakeMichau for support, testing and features.
- @QM for testing and access to games.
- @TheRazerMD for testing and support.
- @Cryio, @krispy, @krisshietala, @Lordubuntu, @scz and @Veeqo for the earlier compatibility matrix.
- The DLSS2FSR community for its support.

This project uses [FreeType](https://gitlab.freedesktop.org/freetype/freetype), licensed under the [FTL](https://gitlab.freedesktop.org/freetype/freetype/-/blob/master/docs/FTL.TXT). Other notices are in [Licenses](../Licenses).

## Upstream sponsorship

Upstream credits [SignPath.io](https://signpath.io/) for Windows code signing and the [SignPath Foundation](https://signpath.org/) for its certificate.

Support upstream: [cdozdil](https://github.com/sponsors/cdozdil?frequency=one-time) and [nitec](https://buymeacoffee.com/nitec).
