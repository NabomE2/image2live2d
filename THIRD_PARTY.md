# Third-party components and usage conditions

## Scope of this change

The repository's [LICENSE](LICENSE) declares Apache-2.0. This change retains that
license. It does not grant rights to artwork, model weights, proprietary runtimes,
or third-party research assets. No weights, SDK/Core binaries, reference Live2D
characters, credentials, or license bypasses are added to the repository or ZIPs.
ZIPs contain the generated model and its referenced texture/JSON files only.

## See-through code and model weights

The GPU Dockerfile pins See-through code to
[`58a1cb11d13f85acec9bbddb8cd4b6487843d4cf`](https://github.com/shitagaki-lab/see-through/tree/58a1cb11d13f85acec9bbddb8cd4b6487843d4cf).
Its [LICENSE at that revision](https://github.com/shitagaki-lab/see-through/blob/58a1cb11d13f85acec9bbddb8cd4b6487843d4cf/LICENSE)
was checked and declares Apache-2.0. Preserve its license/notices when distributing
code or container images. Upstream dependencies have their own terms.

The existing model installer downloads these two repositories directly from
Hugging Face and records the resolved revisions:

- [LayerDiff weights and model card](https://huggingface.co/layerdifforg/seethroughv0.0.2_layerdiff3d)
- [Depth/Marigold weights and model card](https://huggingface.co/24yearsold/seethroughv0.0.1_marigold)

**Weight license verification is incomplete:** the development environment's
egress proxy returned HTTP 403 for Hugging Face metadata requests. The code license
does not establish the weight licenses or underlying base-model conditions.
Review the current cards and all applicable base-model terms before downloading,
using, or redistributing weights. No commercial-use clearance is claimed.
The installer neither bypasses access restrictions nor embeds credentials.
This change does not redistribute weights or change the model installation flow.

## Live2D and VTube Studio

The exporter uses this repository's MOC3 v3.00 writer. The writer's existing source
attributes format research to OpenL2D/moc3ingbird/moc3.hexpat and describes an FDPL
license; that upstream provenance and any downstream obligations require further
review before distribution. That comment alone is not license clearance.

Live2D Cubism Core/SDK and VTube Studio are separately licensed products. Consult
the [official Live2D license information](https://www.live2d.com/en/terms/)
and the terms of the particular SDK/runtime/product you use. Do not assume the
repository's previous comments about research or commercial use establish permission.
This change does not bundle or modify Cubism Core, bypass licensing, or copy
proprietary sample models. Existing optional browser rendering can load a user-supplied
Core or the upstream CDN; those terms still apply.

Generated models are **structurally checked with the repository's own codec**.
Official Cubism Core rendering and Windows VTube Studio import/tracking/physics
are **UNVERIFIED** in this environment. ZIP availability is not certification or
a claim of Live2D endorsement.

Use only artwork you own or have permission to transform and use. Review any
additional conditions of model-generated layers and intended distribution.
