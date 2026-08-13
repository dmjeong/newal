# Third-party notices

newal itself is MIT licensed (see `LICENSE`). It does not bundle or redistribute
any of the packages below — they are declared as dependencies and installed from
PyPI by the user, so each one is distributed by its own project.

Licenses were read from the installed package metadata; versions are the ones
present when this file was last updated. Verify against your own lockfile before
relying on it for a compliance review.

## Required dependencies

| Package | Version | License |
|---|---|---|
| openai (Python SDK) | 3.0.0 | Apache-2.0 |
| httpx | 0.28.1 | BSD-3-Clause |
| pydantic | 2.13.4 | MIT |
| PyYAML | 6.0.1 | MIT |
| rich | 15.0.0 | MIT |
| typer | 0.27.1 | MIT |
| Pillow | 12.3.0 | MIT-CMU (HPND) |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |

`certifi` arrives transitively through httpx and is MPL-2.0, a file-level
copyleft that imposes no obligation on code that merely uses it unmodified.

The `openai` package is an HTTP client for the OpenAI-compatible API shape.
newal points it at a local vLLM or SGLang server; no OpenAI service is involved.

## Optional dependencies

| Extra | Package | Version | License |
|---|---|---|---|
| `video` | opencv-python-headless | 5.0.0.93 | Apache-2.0 (OpenCV) |
| `dev` | pytest | 9.1.1 | MIT |
| `dev` | ruff | 0.16.2 | MIT |

### A note on the `video` extra

OpenCV itself is Apache-2.0. The PyPI wheels additionally bundle **FFmpeg**,
which the wheel's own `LICENSE-3RD-PARTY.txt` states plainly:

> FFmpeg is redistributed within all opencv-python packages.

That FFmpeg is built **without** GPL-only components — no x264, x265,
libpostproc, or frei0r are present in the wheel's third-party licenses — so it
is under the LGPL (2.1 / 3), not the GPL.

Because newal only declares the dependency and never ships the wheel, the LGPL
obligations attach to the opencv-python project rather than to this repository.
They would become relevant to a downstream user who redistributes an artifact
containing `cv2` — a container image, a frozen installer, a vendored wheel. Note
in that case that the wheel links FFmpeg *statically* into a single shared
object, which under the LGPL carries a heavier relinking obligation than dynamic
linking does.

The `video` extra is optional. Without it newal still handles text and images;
only video attachments are unavailable.

## Not verified here

These are declared in `pyproject.toml` extras but were not installed in the
environment where this file was generated, so their licenses were not read from
metadata. Check them yourself before depending on them:

`transformers`, `torch`, `accelerate` (the `transformers` extra) and
`sentence-transformers` (the `dense` extra).

The inference engines newal talks to — vLLM and SGLang — are installed
separately by the user and are not dependencies of this package.

## Model weights

Model weights are downloaded by the user at runtime and are **not** covered by
this project's license. Qwen licensing varies by model: many are Apache-2.0, but
some carry additional terms. Check the model card for every model you enable in
`configs/`, and check the base model's terms before distributing anything
fine-tuned from it.
