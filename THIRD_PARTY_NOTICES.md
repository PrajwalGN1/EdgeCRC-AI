# Third-Party Software & Dependency Notices (`THIRD_PARTY_NOTICES.md`)

This document records verified third-party software libraries, build toolchains, and upstream foundation models referenced or linked by **EdgeCRC-AI**.

- **License Retention:** Every third-party component listed below retains its own upstream copyright and license. Nothing in the root [`LICENSE`](LICENSE) (**PolyForm Noncommercial License 1.0.0**) replaces, modifies, or relicenses third-party components.
- **Repository Distribution Footprint:** This Git repository distributes source code and project assets only; Python wheels and Android `.aar`/`.jar` dependencies are resolved dynamically from PyPI, Google Maven, and Maven Central at build/install time (except the standard Gradle Wrapper bootstrap `android/gradle/wrapper/gradle-wrapper.jar` under `Apache-2.0`). If a compiled Android `.apk` binary is built and distributed in compliance with [`MODEL_LICENSE.md`](MODEL_LICENSE.md), runtime `implementation` notices in Section 1 and Section 5 must accompany that binary distribution.

---

## 1. Android Application & Gradle Build Dependencies (`android/`)

Inspected directly from `android/build.gradle.kts`, `android/app/build.gradle.kts`, and `android/gradle/wrapper/gradle-wrapper.properties`:

| Component | Version | Scope | Upstream Project | Verified License | Official Source | Redistribution Obligations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `com.microsoft.onnxruntime:onnxruntime-android` | `1.18.0` | Runtime (`implementation`) | Microsoft ONNX Runtime | `MIT` | https://github.com/microsoft/onnxruntime | Include MIT copyright and permission notice when distributing compiled APK binaries (see Section 5.1). |
| `androidx.core:core-ktx` | `1.13.1` | Runtime (`implementation`) | Android Open Source Project (`AndroidX`) | `Apache-2.0` | https://developer.android.com/jetpack/androidx/releases/core | Include Apache-2.0 license notice and any upstream `NOTICE` attributions when distributing binaries. |
| `androidx.appcompat:appcompat` | `1.6.1` | Runtime (`implementation`) | Android Open Source Project (`AndroidX`) | `Apache-2.0` | https://developer.android.com/jetpack/androidx/releases/appcompat | Include Apache-2.0 license notice when distributing binaries. |
| `com.google.android.material:material` | `1.12.0` | Runtime (`implementation`) | Material Components for Android (Google) | `Apache-2.0` | https://github.com/material-components/material-components-android | Include Apache-2.0 license notice when distributing binaries. |
| `org.jetbrains.kotlin.android` / `kotlin-stdlib` | `1.9.24` | Build Plugin & Runtime | JetBrains Kotlin | `Apache-2.0` | https://github.com/JetBrains/kotlin | Include Apache-2.0 license notice when distributing binaries. |
| `com.android.application` (Android Gradle Plugin) | `8.4.2` | Build Plugin (`apply false`) | Android Studio / AOSP Build Tools | `Apache-2.0` | https://developer.android.com/build/releases/gradle-plugin | Build-time plugin only; not packaged inside APK. |
| Gradle Wrapper (`gradle-wrapper.jar`) | `8.10.2` | Build Wrapper (`android/gradle/wrapper/`) | Gradle Inc. | `Apache-2.0` | https://github.com/gradle/gradle | Retain Apache-2.0 license reference when redistributing repository source containing `gradle-wrapper.jar`. |
| `junit:junit` | `4.13.2` | Host Unit Test (`testImplementation`) | JUnit Team | `EPL-1.0` | https://github.com/junit-team/junit4 | Host JVM unit test scope only; **not** packaged in debug or release APKs. |
| `org.jetbrains.kotlin:kotlin-test` | `1.9.24` | Test (`testImplementation` & `androidTestImplementation`) | JetBrains Kotlin | `Apache-2.0` | https://github.com/JetBrains/kotlin | Test scope only; not packaged in main application APK. |
| `com.microsoft.onnxruntime:onnxruntime` | `1.18.0` | Host Unit Test (`testImplementation`) | Microsoft ONNX Runtime (JVM Desktop) | `MIT` | https://github.com/microsoft/onnxruntime | Host JVM parity test scope only; not packaged in Android APK. |
| `org.json:json` | `20240303` | Host Unit Test (`testImplementation`) | JSON-java (Douglas Crockford / Sean Leary) | `JSON` (MIT-variant with *"The Software shall be used for Good, not Evil"*) | https://github.com/stleary/JSON-java | Host JVM unit test scope only (Android runtime uses AOSP clean-room `org.json` SDK classes under `Apache-2.0`); **not** packaged in APK. |
| `androidx.test:runner` | `1.5.2` | Instrumentation Test (`androidTestImplementation`) | AndroidX Test | `Apache-2.0` | https://developer.android.com/jetpack/androidx/releases/test | Test APK only; not packaged in main APK. |
| `androidx.test.ext:junit` | `1.1.5` | Instrumentation Test (`androidTestImplementation`) | AndroidX Test | `Apache-2.0` | https://developer.android.com/jetpack/androidx/releases/test | Test APK only; not packaged in main APK. |

---

## 2. Upstream Foundation Models Referenced by Python Pipeline

Detailed legal terms, `Results` classifications, and ShareAlike obligations are documented in [`MODEL_LICENSE.md`](MODEL_LICENSE.md):

| Component | Version / Identifier | Upstream Project | Verified License | Official Source | Redistribution Obligations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Owkin Phikon-v2** | `owkin/phikon-v2` (`ViT-L/16`) | Owkin Inc. | **Owkin Phikon-v2 License Terms and Conditions** (`LICENSE.pdf`) | https://huggingface.co/owkin/phikon-v2 | Non-commercial research by non-profit entities only; downstream `Results` (`models/*`, `reference_tensors/*`) subject to ShareAlike under Owkin's terms without additional/conflicting restrictions; commercial use requires Owkin Inc. agreement. |
| **Bio_ClinicalBERT** | `emilyalsentzer/Bio_ClinicalBERT` | Emily Alsentzer et al. (MIT CSAIL) | `MIT` | https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT & https://github.com/EmilyAlsentzer/clinicalBERT | Retain copyright and MIT permission notice (`Copyright (c) 2019 Emily Alsentzer`) and academic citation (Alsentzer et al., 2019). |

---

## 3. Pinned Python Dependencies (`requirements.txt`)

All `75` pinned packages in `requirements.txt` were inspected directly against installed Python package distribution metadata (`importlib.metadata` `License`, `License-Expression`, and `Classifier` fields). None of these packages are vendored or redistributed inside this repository:

| Component | Version | Upstream Project | Verified License | Official Source | Redistribution Obligations (If Bundled/Redistributed) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `aiohappyeyeballs` | `2.6.1` | aio-libs | `PSF-2.0` | https://github.com/aio-libs/aiohappyeyeballs | Retain PSF-2.0 copyright and license notice. |
| `aiohttp` | `3.13.2` | aio-libs | `Apache-2.0 AND MIT` | https://github.com/aio-libs/aiohttp | Retain Apache-2.0 and MIT copyright/license notices. |
| `aiosignal` | `1.4.0` | aio-libs | `Apache-2.0` | https://github.com/aio-libs/aiosignal | Retain Apache-2.0 license and NOTICE text. |
| `altair` | `6.0.0` | Vega-Altair Developers | `BSD-3-Clause` | https://github.com/vega/altair | Retain BSD copyright notice and disclaimer. |
| `anyio` | `4.12.0` | Alex Grönholm / anyio | `MIT` | https://github.com/agronholm/anyio | Retain MIT copyright and permission notice. |
| `attrs` | `25.4.0` | Hynek Schlawack / python-attrs | `MIT` | https://github.com/python-attrs/attrs | Retain MIT copyright and permission notice. |
| `blinker` | `1.9.0` | Pallets Ecosystem | `MIT` | https://github.com/pallets-eco/blinker | Retain MIT copyright and permission notice. |
| `cachetools` | `6.2.4` | Thomas Kemmer | `MIT` | https://github.com/tkem/cachetools | Retain MIT copyright and permission notice. |
| `certifi` | `2025.11.12` | Certifi / Mozilla CA Bundle | `MPL-2.0` | https://github.com/certifi/python-certifi | File-level weak copyleft if modified/redistributed; make MPL-2.0 source notices available for certifi files. |
| `charset-normalizer` | `3.4.4` | Ahmed Tahri (`Ousret`) | `MIT` | https://github.com/jawah/charset_normalizer | Retain MIT copyright and permission notice. |
| `click` | `8.3.1` | Pallets | `BSD-3-Clause` | https://click.palletsprojects.com/ | Retain BSD-3-Clause copyright notice and disclaimer. |
| `filelock` | `3.20.1` | tox-dev | `Unlicense` | https://github.com/tox-dev/py-filelock | Public-domain dedication (`Unlicense`). |
| `frozenlist` | `1.8.0` | aio-libs | `Apache-2.0` | https://github.com/aio-libs/frozenlist | Retain Apache-2.0 license notice. |
| `fsspec` | `2025.12.0` | Filesystem Spec (`fsspec`) | `BSD-3-Clause` | https://github.com/fsspec/filesystem_spec | Retain BSD-3-Clause copyright notice and disclaimer. |
| `gitdb` | `4.0.12` | GitPython Developers | `BSD-3-Clause` | https://github.com/gitpython-developers/gitdb | Retain BSD copyright notice and disclaimer. |
| `GitPython` | `3.1.45` | GitPython Developers | `BSD-3-Clause` | https://github.com/gitpython-developers/GitPython | Retain BSD-3-Clause copyright notice and disclaimer. |
| `greenlet` | `3.2.4` | python-greenlet | `MIT AND PSF-2.0` | https://github.com/python-greenlet/greenlet | Retain MIT and PSF-2.0 copyright/license notices. |
| `h11` | `0.16.0` | python-hyper | `MIT` | https://github.com/python-hyper/h11 | Retain MIT copyright and permission notice. |
| `h5py` | `3.15.1` | h5py Project | `BSD-3-Clause` | https://github.com/h5py/h5py | Retain BSD-3-Clause copyright notice and HDF5 notices if binary wheels redistributed. |
| `hf-xet` | `1.2.0` | Hugging Face (`xet-core`) | `Apache-2.0` | https://github.com/huggingface/xet-core | Retain Apache-2.0 license notice. |
| `httpcore` | `1.0.9` | Encode | `BSD-3-Clause` | https://github.com/encode/httpcore | Retain BSD-3-Clause copyright notice and disclaimer. |
| `httpx` | `0.28.1` | Encode | `BSD-3-Clause` | https://github.com/encode/httpx | Retain BSD-3-Clause copyright notice and disclaimer. |
| `huggingface_hub` | `1.2.3` | Hugging Face | `Apache-2.0` | https://github.com/huggingface/huggingface_hub | Retain Apache-2.0 license notice. |
| `idna` | `3.11` | Kim Davies | `BSD-3-Clause` | https://github.com/kjd/idna | Retain BSD-3-Clause copyright notice and Unicode table notices. |
| `Jinja2` | `3.1.6` | Pallets | `BSD-3-Clause` | https://jinja.palletsprojects.com/ | Retain BSD-3-Clause copyright notice and disclaimer. |
| `jsonschema` | `4.25.1` | python-jsonschema | `MIT` | https://github.com/python-jsonschema/jsonschema | Retain MIT copyright and permission notice. |
| `jsonschema-specifications` | `2025.9.1` | python-jsonschema | `MIT` | https://github.com/python-jsonschema/jsonschema-specifications | Retain MIT copyright and permission notice. |
| `lightning-utilities` | `0.15.2` | Lightning AI | `Apache-2.0` | https://github.com/Lightning-AI/utilities | Retain Apache-2.0 license notice. |
| `MarkupSafe` | `3.0.3` | Pallets | `BSD-3-Clause` | https://markupsafe.palletsprojects.com/ | Retain BSD-3-Clause copyright notice and disclaimer. |
| `monai` | `1.5.1` | Project MONAI | `Apache-2.0` | https://github.com/Project-MONAI/MONAI | Retain Apache-2.0 license and NOTICE file. |
| `mpmath` | `1.3.0` | Fredrik Johansson / mpmath | `BSD-3-Clause` | https://github.com/fredrik-johansson/mpmath | Retain BSD copyright notice and disclaimer. |
| `multidict` | `6.7.0` | aio-libs | `Apache-2.0` | https://github.com/aio-libs/multidict | Retain Apache-2.0 license notice. |
| `narwhals` | `2.14.0` | Marco Gorelli / narwhals-dev | `MIT` | https://github.com/narwhals-dev/narwhals | Retain MIT copyright and permission notice. |
| `networkx` | `3.6.1` | NetworkX Developers | `BSD-3-Clause` | https://github.com/networkx/networkx | Retain BSD-3-Clause copyright notice and disclaimer. |
| `numpy` | `2.2.6` | NumPy Developers | `BSD-3-Clause` | https://github.com/numpy/numpy | Retain BSD-3-Clause copyright notice (and bundled OpenBLAS notices if wheels redistributed). |
| `opencv-python` | `4.12.0.88` | OpenCV / opencv-python | `Apache-2.0` | https://github.com/opencv/opencv-python | Retain Apache-2.0 license notice (and third-party codec notices if binary wheels redistributed). |
| `packaging` | `25.0` | PyPA | `Apache-2.0 OR BSD-2-Clause` | https://github.com/pypa/packaging | Retain either Apache-2.0 or BSD-2-Clause notice. |
| `pandas` | `2.3.3` | pandas Development Team | `BSD-3-Clause` | https://github.com/pandas-dev/pandas | Retain BSD-3-Clause copyright notice and disclaimer. |
| `pillow` | `12.0.0` | Python Pillow (PIL Fork) | `MIT-CMU` | https://github.com/python-pillow/Pillow | Retain MIT-CMU Historical Permission Notice and Disclaimer. |
| `playwright` | `1.56.0` | Microsoft Playwright Python | `Apache-2.0` | https://github.com/microsoft/playwright-python | Retain Apache-2.0 license and NOTICE text. |
| `propcache` | `0.4.1` | aio-libs | `Apache-2.0` | https://github.com/aio-libs/propcache | Retain Apache-2.0 license notice. |
| `protobuf` | `6.33.2` | Google Protocol Buffers | `BSD-3-Clause` | https://developers.google.com/protocol-buffers/ | Retain BSD-3-Clause copyright notice and disclaimer. |
| `pyarrow` | `22.0.0` | Apache Arrow | `Apache-2.0` | https://github.com/apache/arrow | Retain Apache-2.0 license and NOTICE file. |
| `pydeck` | `0.9.1` | vis.gl / deck.gl | `Apache-2.0` | https://github.com/visgl/deck.gl/tree/master/bindings/pydeck | Retain Apache-2.0 license notice. |
| `pyee` | `13.0.0` | Joshua Holbrook | `MIT` | https://github.com/jfhbrook/pyee | Retain MIT copyright and permission notice. |
| `python-dateutil` | `2.9.0.post0` | dateutil | `Apache-2.0 OR BSD-3-Clause` | https://github.com/dateutil/dateutil | Retain Apache-2.0 or BSD-3-Clause copyright/license notice. |
| `pytorch-lightning` | `2.6.0` | Lightning AI | `Apache-2.0` | https://github.com/Lightning-AI/pytorch-lightning | Retain Apache-2.0 license and NOTICE file. |
| `pytz` | `2025.2` | Stuart Bishop | `MIT` | https://pythonhosted.org/pytz/ | Retain MIT copyright and permission notice. |
| `PyYAML` | `6.0.3` | YAML / PyYAML | `MIT` | https://github.com/yaml/pyyaml | Retain MIT copyright and permission notice. |
| `referencing` | `0.37.0` | python-jsonschema | `MIT` | https://github.com/python-jsonschema/referencing | Retain MIT copyright and permission notice. |
| `requests` | `2.32.5` | PSF / Kenneth Reitz | `Apache-2.0` | https://github.com/psf/requests | Retain Apache-2.0 license and NOTICE text. |
| `rpds-py` | `0.30.0` | Julian Berman / crate-py | `MIT` | https://github.com/crate-py/rpds | Retain MIT copyright and permission notice. |
| `safetensors` | `0.7.0` | Hugging Face | `Apache-2.0` | https://github.com/huggingface/safetensors | Retain Apache-2.0 license notice. |
| `segmentation_models_pytorch` | `0.5.0` | Pavel Iakubovskii / qubvel-org | `MIT` | https://github.com/qubvel-org/segmentation_models.pytorch | Retain MIT copyright and permission notice. |
| `setuptools` | `80.9.0` | PyPA | `MIT` | https://github.com/pypa/setuptools | Retain MIT copyright and permission notice. |
| `shellingham` | `1.5.4` | Tzu-ping Chung (`uranusjr`) | `ISC` | https://github.com/sarugaku/shellingham | Retain ISC copyright and permission notice. |
| `six` | `1.17.0` | Benjamin Peterson | `MIT` | https://github.com/benjaminp/six | Retain MIT copyright and permission notice. |
| `smmap` | `5.0.2` | GitPython Developers | `BSD-3-Clause` | https://github.com/gitpython-developers/smmap | Retain BSD-3-Clause copyright notice and disclaimer. |
| `streamlit` | `1.52.2` | Snowflake / Streamlit Inc. | `Apache-2.0` | https://github.com/streamlit/streamlit | Retain Apache-2.0 license and NOTICE file. |
| `sympy` | `1.14.0` | SymPy Development Team | `BSD-3-Clause` | https://github.com/sympy/sympy | Retain BSD-3-Clause copyright notice and disclaimer. |
| `tenacity` | `9.1.2` | Julien Danjou | `Apache-2.0` | https://github.com/jd/tenacity | Retain Apache-2.0 license notice. |
| `timm` | `1.0.22` | Ross Wightman / Hugging Face | `Apache-2.0` | https://github.com/huggingface/pytorch-image-models | Retain Apache-2.0 license notice (individual third-party pretrained model weights in `timm` have separate upstream terms if used). |
| `toml` | `0.10.2` | William Pearson (`uiri`) | `MIT` | https://github.com/uiri/toml | Retain MIT copyright and permission notice. |
| `torch` | `2.9.1` | PyTorch Foundation / Meta | `BSD-3-Clause` | https://github.com/pytorch/pytorch | Retain BSD-3-Clause copyright/license notice (CUDA/MKL binary wheels carry separate vendor runtime terms if redistributed). |
| `torchaudio` | `2.9.1` | PyTorch Foundation | `BSD-2-Clause` | https://github.com/pytorch/audio | Retain BSD copyright notice and disclaimer. |
| `torchmetrics` | `1.8.2` | Lightning AI | `Apache-2.0` | https://github.com/Lightning-AI/torchmetrics | Retain Apache-2.0 license notice. |
| `torchvision` | `0.24.1` | PyTorch Foundation | `BSD-3-Clause` | https://github.com/pytorch/vision | Retain BSD-3-Clause copyright notice and disclaimer. |
| `tornado` | `6.5.4` | Tornado Web / Facebook | `Apache-2.0` | https://github.com/tornadoweb/tornado | Retain Apache-2.0 license notice. |
| `tqdm` | `4.67.1` | tqdm Developers | `MPL-2.0 AND MIT` | https://github.com/tqdm/tqdm | Retain MIT notice and MPL-2.0 file-level notice if redistributed. |
| `typer-slim` | `0.21.0` | Sebastián Ramírez (`tiangolo`) | `MIT` | https://github.com/fastapi/typer | Retain MIT copyright and permission notice. |
| `typing_extensions` | `4.15.0` | Python Typing / PSF | `PSF-2.0` | https://github.com/python/typing_extensions | Retain PSF-2.0 copyright and license notice. |
| `tzdata` | `2025.3` | Python / IANA Time Zone Database | `Apache-2.0` | https://github.com/python/tzdata | Retain Apache-2.0 wrapper notice (IANA data is public domain). |
| `urllib3` | `2.6.2` | Andrey Petrov / urllib3 | `MIT` | https://github.com/urllib3/urllib3 | Retain MIT copyright and permission notice. |
| `yarl` | `1.22.0` | aio-libs | `Apache-2.0` | https://github.com/aio-libs/yarl | Retain Apache-2.0 license notice. |
| `yt-dlp` | `2025.12.8` | yt-dlp Contributors *(unused dev-env freeze entry)* | `Unlicense` | https://github.com/yt-dlp/yt-dlp | Public-domain dedication (`Unlicense`); not imported by EdgeCRC-AI. |

---

## 4. Additional Python Export, Evaluation & Test Toolchain Packages

The following packages are imported by `src/`, `tools/`, or `tests/` during upstream feature extraction, ONNX graph export, benchmark plotting, or unit testing (verified from the active local test environment):

| Component | Verified Environment Version | Upstream Project | Verified License | Official Source | Redistribution Obligations (If Bundled/Redistributed) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `onnx` | `1.23.2` | Linux Foundation AI & Data (`ONNX`) | `Apache-2.0` | https://github.com/onnx/onnx | Retain Apache-2.0 license notice. |
| `onnxruntime` | `1.31.0` (Python) / `1.18.0` (Android) | Microsoft ONNX Runtime | `MIT` | https://github.com/microsoft/onnxruntime | Retain MIT copyright and permission notice (see Section 5.1). |
| `transformers` | `4.38.2` | Hugging Face | `Apache-2.0` | https://github.com/huggingface/transformers | Retain Apache-2.0 license notice. |
| `scikit-learn` | `1.6.1` | scikit-learn Developers | `BSD-3-Clause` | https://github.com/scikit-learn/scikit-learn | Retain BSD-3-Clause copyright notice and disclaimer. |
| `scipy` | `1.16.0` | SciPy Developers | `BSD-3-Clause` | https://github.com/scipy/scipy | Retain BSD-3-Clause copyright notice and disclaimer. |
| `matplotlib` | `3.9.2` | Matplotlib Development Team | `PSF-based` (Matplotlib License) | https://matplotlib.org/stable/project/license.html | Retain Matplotlib license agreement and copyright notice. |
| `pytest` | `9.1.1` | Holger Krekel / pytest-dev | `MIT` | https://github.com/pytest-dev/pytest | Test runner only; retain MIT notice if redistributed. |

---

## 5. Key Runtime License Notices (For Binary APK Builds)

### 5.1 Microsoft ONNX Runtime (`MIT License`)

```text
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 5.2 Apache License, Version 2.0 (`AndroidX`, `Material Components`, `Kotlin`, `Gradle`)

AndroidX (`core-ktx`, `appcompat`), Google Material Components (`com.google.android.material:material`), JetBrains Kotlin (`kotlin-stdlib`), and the Gradle Wrapper are licensed under the **Apache License, Version 2.0**:
- Full official license text: https://www.apache.org/licenses/LICENSE-2.0
