# Raidionics backend for segmentation/classification

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Build Actions Status](https://github.com/dbouget/raidionics-seg-lib/workflows/Build/badge.svg)](https://github.com/dbouget/raidionics-seg-lib/actions)
[![Paper](https://zenodo.org/badge/DOI/10.3389/fneur.2022.932219.svg)](https://www.frontiersin.org/articles/10.3389/fneur.2022.932219/full)

The code corresponds to the segmentation or classification backend of MRI/CT volumes, using ONNX runtime for inference.  
The module can either be used as a Python library, as CLI, or as Docker container.

# Installation

```
pip install git+https://github.com/dbouget/raidionics-seg-lib.git
```

By default, inference is performed on CPU only.

# Usage
## CLI
```
raidionicsseg CONFIG
```

CONFIG should point to a configuration file (*.ini), specifying all runtime parameters,
according to the pattern from [**blank_main_config.ini**](https://github.com/dbouget/raidionics-seg-lib/blob/master/blank_main_config.ini).

## Python module
```
from raidionicsseg import run_model
run_model(config_filename="/path/to/main_config.ini")
```

## Docker
:warning: The Docker image can only perform inference using the CPU, there is no GPU support at this stage.
```
docker pull dbouget/raidionics-segmenter:v1.2
docker run --entrypoint /bin/bash -v /home/<username>/<resources_path>:/home/ubuntu/resources -t -i --runtime=nvidia --network=host --ipc=host dbouget/raidionics-segmenter:v1.2
```

The `/home/<username>/<resources_path>` before the column sign has to be changed to match a directory on your local 
machine containing the data to expose to the docker image. Namely, it must contain folder(s) with images you want to 
run inference on, as long as a folder with the trained models to use, and a destination folder where the results will 
be placed.

# Models
The trained models are automatically downloaded when running Raidionics or Raidionics-Slicer.  
Alternatively, all existing Raidionics models can be browsed [here](https://drive.google.com/drive/folders/1TUTPzY73Kyt4WedqHpeM6i-gmfOfwqWW?usp=sharing), 
i.e., old and new models, unit test resources.

# Developers
For running inference on GPU, your machine must be properly configured (cf. [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html))  

To run the unit tests, type the following within your virtual environment and within the raidionics-seg-lib folder:
```
pip install pytest
pytest tests/
```

# How to cite
If you are using Raidionics in your research, please use the following citation:
```
@article{10.3389/fneur.2022.932219,
title={Preoperative Brain Tumor Imaging: Models and Software for Segmentation and Standardized Reporting},
author={Bouget, David and Pedersen, André and Jakola, Asgeir S. and Kavouridis, Vasileios and Emblem, Kyrre E. and Eijgelaar, Roelant S. and Kommers, Ivar and Ardon, Hilko and Barkhof, Frederik and Bello, Lorenzo and Berger, Mitchel S. and Conti Nibali, Marco and Furtner, Julia and Hervey-Jumper, Shawn and Idema, Albert J. S. and Kiesel, Barbara and Kloet, Alfred and Mandonnet, Emmanuel and Müller, Domenique M. J. and Robe, Pierre A. and Rossi, Marco and Sciortino, Tommaso and Van den Brink, Wimar A. and Wagemakers, Michiel and Widhalm, Georg and Witte, Marnix G. and Zwinderman, Aeilko H. and De Witt Hamer, Philip C. and Solheim, Ole and Reinertsen, Ingerid},
journal={Frontiers in Neurology},
volume={13},
year={2022},
url={https://www.frontiersin.org/articles/10.3389/fneur.2022.932219},
doi={10.3389/fneur.2022.932219},
issn={1664-2295}}
```
