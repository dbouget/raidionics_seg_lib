<div align="center">

# Raidionics Segmentation Backend

**Segmentation and classification library for MRI/CT volumes, powered by ONNX Runtime.**

Use it as a Python package, a CLI tool, or a Docker container — backend engine behind [Raidionics](https://github.com/raidionics/Raidionics) and [Raidionics-Slicer](https://github.com/raidionics/Raidionics-Slicer).

[![PyPI version](https://img.shields.io/pypi/v/raidionicsseg.svg)](https://pypi.org/project/raidionicsseg/)
[![Python](https://img.shields.io/badge/python-3.9%7C3.10%7C3.11%7C3.12%7C3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![codecov](https://img.shields.io/codecov/c/github/dbouget/raidionics_seg_lib)](https://codecov.io/gh/dbouget/raidionics_seg_lib)
[![Paper](https://img.shields.io/badge/DOI-10.3389%2Ffneur.2022.932219-blue.svg)](https://www.frontiersin.org/articles/10.3389/fneur.2022.932219/full)

</div>

---

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
  - [CLI](#cli)
  - [Python module](#python-module)
  - [Docker](#docker)
- [Models](#models)
- [GPU support](#gpu-support)
- [Development](#development)
- [How to cite](#how-to-cite)
- [License](#license)

---

## Overview

This library provides the inference backend for segmenting and classifying central nervous system tumors (and related structures) in MRI/CT volumes. It runs on **ONNX Runtime** by default (CPU-only), with optional GPU acceleration via `onnxruntime-gpu` or PyTorch.

It is designed to be used in three ways:

| Mode | Best for |
|---|---|
| **Python module** | Integrating segmentation into your own pipeline |
| **CLI** | Quick, scriptable inference from a config file |
| **Docker** | Reproducible environments, no local Python setup needed |

---

## Installation

```bash
pip install raidionicsseg
```

Or install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/dbouget/raidionics_seg_lib.git
```

**Optional extras** (only needed for GPU inference):

```bash
pip install raidionicsseg[ort-gpu]   # ONNX Runtime GPU
pip install raidionicsseg[torch]     # PyTorch backend
```

---

## Quick start

1. Copy [`blank_main_config.ini`](blank_main_config.ini) and fill in your input/output paths and model selection.
2. Run inference:

```bash
raidionicsseg /path/to/your_config.ini
```

That's it — see [Usage](#usage) below for the Python API and Docker equivalents.

---

## Usage

### CLI

```bash
raidionicsseg CONFIG
```

`CONFIG` is a path to an `.ini` file specifying all runtime parameters, following the structure in [`blank_main_config.ini`](blank_main_config.ini).

### Python module

```python
from raidionicsseg import run_model

run_model(config_filename="/path/to/main_config.ini")
```

### Docker

```bash
docker pull dbouget/raidionics-segmenter:v1.5.0-py39-cpu

docker run \
  -v /home/<username>/<resources_path>:/workspace/resources \
  -t -i --network=host --ipc=host --user $(id -u) \
  dbouget/raidionics-segmenter:v1.5.0-py39-cpu \
  -c /workspace/resources/<path>/<to>/main_config.ini -v <verbose>
```

This runs CPU-only inference. For GPU support, an interactive shell, path-mapping details, and troubleshooting, see the full **[Docker guide](docs/docker.md)**.

---

## Models

Trained models are downloaded automatically when running Raidionics or Raidionics-Slicer. To browse all available models directly, see the [Raidionics-models](https://github.com/dbouget/Raidionics-models) repository.

---

## GPU support

To run inference on GPU:

1. Configure your machine per the [ONNX Runtime CUDA execution provider guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
2. Install `onnxruntime-gpu` matching your driver/CUDA version ([compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cuda-12x)).
3. Set the `gpu_id` parameter in your configuration file to the target GPU.

---

## Development

Run the test suite from within the repository root and your virtual environment:

```bash
pip install pytest
pytest tests/
```

---

## How to cite

If you use Raidionics in your research, please cite the software and the associated papers. Citation metadata is provided in [`CITATION.cff`](CITATION.cff) — click **"Cite this repository"** in the sidebar for ready-to-use APA/BibTeX formats, covering both the main software release (Scientific Reports, 2023) and the preliminary validation study (Frontiers in Neurology, 2022).

---

## License

Distributed under the [BSD-2-Clause License](LICENSE.md).