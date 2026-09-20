# Provenance and Licenses

The source inventory is recorded in
`src/retriever_src_kitchen/_scene/asset-manifest.json`. Hashes identify the local
snapshot; they do not establish permission to publish it.

| Component | Source | Retained Notice |
| --- | --- | --- |
| Panda, UR5e, TIAGo models | robosuite 1.5.2 exported robot descriptions and referenced assets | `_scene/output/robots/LICENSE.robosuite` |
| Mobile ALOHA | [AgileX mobile_aloha_sim](https://github.com/agilexrobotics/mobile_aloha_sim/tree/2843ff11d2695c1563a1c9847f632aa9734f5bbc/aloha_isaac_sim), pinned revision in the build helper | `_scene/vendor/mobile_aloha/LICENSE` (MIT) |
| Mink source | mink 0.0.13 | `_scene/vendor/motion/mink-0.0.13.dist-info/licenses/LICENSE` (Apache-2.0) |
| Preview data models, controls and console helper | GoldenRetriever RoboCasa examples: `embodied.py`, `runtime.py`, `web_console.py` | `_scene/preview_support/LICENSE` (Apache-2.0) |
| Lucide browser icons | lucide 0.468.0 | `_scene/vendor/LICENSE.lucide` (ISC) |
| Browser recording rasterizer | html2canvas 1.4.1 | `_scene/vendor/LICENSE.html2canvas` (MIT) |

Robot XML asset paths were made relative. Mobile ALOHA was exported from the
upstream URDF in a parked pose, then converted into visual-only MJCF geometry.
Authoring-machine metadata was removed from PNGs with decoded pixels unchanged.
Its geometry has no joints, collision participation, or actuators in this scene.
It is the AgileX simulation variant and must not be described as a calibrated
digital twin of Stanford's original Mobile ALOHA.

The preview-support imports were relocated; simulator startup and local port
selection belong to this application. No model-provider use is enabled by the
standalone preview. Dependency packages retain their installed licenses.

The room geometry, baked textures, and SRC display artwork came from the scene's
authoring inputs. Their inclusion in this private package does not establish
public redistribution rights, trademark permission, or endorsement. The
robosuite notice also does not replace review of individual robot-mesh terms.
