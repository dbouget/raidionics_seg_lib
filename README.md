# Raidionics backend for segmentation and classification

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![](https://img.shields.io/badge/python-3.9|3.10|3.11|3.12|3.13-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://zenodo.org/badge/DOI/10.3389/fneur.2022.932219.svg)](https://www.frontiersin.org/articles/10.3389/fneur.2022.932219/full)
[![codecov](https://codecov.io/gh/dbouget/raidionics_seg_lib/branch/master/graph/badge.svg?token=ZSPQVR7RKX)](https://codecov.io/gh/dbouget/raidionics_seg_lib)
[![PyPI version](https://img.shields.io/pypi/v/raidionicsseg.svg)](https://pypi.org/project/raidionicsseg/)


The code corresponds to the segmentation or classification backend of MRI/CT volumes, using ONNX runtime for inference.  
The module can either be used as a Python library, as CLI, or as Docker container. By default, inference is performed on CPU only.

## [Installation](https://github.com/dbouget/raidionics_seg_lib#installation)

```
pip install raidionicsseg
(or)
pip install git+https://github.com/dbouget/raidionics_seg_lib.git
```

## [How to cite](https://github.com/dbouget/raidionics_seg_lib#how-to-cite)
If you are using Raidionics in your research, please cite the following references.

The final software including updated performance metrics for preoperative tumors and introducing postoperative tumor segmentation:
```
@article{bouget2023raidionics,
    author = {Bouget, David and Alsinan, Demah and Gaitan, Valeria and Holden Helland, Ragnhild and Pedersen, André and Solheim, Ole and Reinertsen, Ingerid},
    year = {2023},
    month = {09},
    pages = {},
    title = {Raidionics: an open software for pre-and postoperative central nervous system tumor segmentation and standardized reporting},
    volume = {13},
    journal = {Scientific Reports},
    doi = {10.1038/s41598-023-42048-7},
}
```

For the preliminary preoperative tumor segmentation validation and software features:
```
@article{bouget2022preoptumorseg,
    title={Preoperative Brain Tumor Imaging: Models and Software for Segmentation and Standardized Reporting},
    author={Bouget, David and Pedersen, André and Jakola, Asgeir S. and Kavouridis, Vasileios and Emblem, Kyrre E. and Eijgelaar, Roelant S. and Kommers, Ivar and Ardon, Hilko and Barkhof, Frederik and Bello, Lorenzo and Berger, Mitchel S. and Conti Nibali, Marco and Furtner, Julia and Hervey-Jumper, Shawn and Idema, Albert J. S. and Kiesel, Barbara and Kloet, Alfred and Mandonnet, Emmanuel and Müller, Domenique M. J. and Robe, Pierre A. and Rossi, Marco and Sciortino, Tommaso and Van den Brink, Wimar A. and Wagemakers, Michiel and Widhalm, Georg and Witte, Marnix G. and Zwinderman, Aeilko H. and De Witt Hamer, Philip C. and Solheim, Ole and Reinertsen, Ingerid},
    journal={Frontiers in Neurology},
    volume={13},
    year={2022},
    url={https://www.frontiersin.org/articles/10.3389/fneur.2022.932219},
    doi={10.3389/fneur.2022.932219},
    issn={1664-2295}
}
```

# [Usage](https://github.com/dbouget/raidionics_seg_lib#usage)

## [1. CLI](https://github.com/dbouget/raidionics_seg_lib#cli)
```
raidionicsseg CONFIG
```

CONFIG should point to a configuration file (*.ini), specifying all runtime parameters,
according to the pattern from [**blank_main_config.ini**](https://github.com/dbouget/raidionics-seg-lib/blob/master/blank_main_config.ini).

## [2. Python module](https://github.com/dbouget/raidionics_seg_lib#python-module)
```
from raidionicsseg import run_model
run_model(config_filename="/path/to/main_config.ini")
```

## [3. Docker](https://github.com/dbouget/raidionics_seg_lib#docker)
When calling Docker images, the --user flag must be properly used in order for the folders and files created inside
the container to inherit the proper read/write permissions. The user ID is retrieved on-the-fly in the following
examples, but it can be given in a more hard-coded fashion if known by the user.

:warning: The following Docker image can only perform inference using the CPU. Another Docker image has been created, able to leverage
the GPU (see further down below). If the CUDA version does not match your machine, a new Docker image can be built manually, 
simply modifying the base torch image to pull from inside Dockerfile_gpu.

```
docker pull dbouget/raidionics-segmenter:v1.4-py39-cpu
```

For opening the Docker image and interacting with it, run:  
```
docker run --entrypoint /bin/bash -v /home/<username>/<resources_path>:/workspace/resources -t -i --network=host --ipc=host --user $(id -u) dbouget/raidionics-segmenter:v1.4-py39-cpu
```

The `/home/<username>/<resources_path>` before the column sign has to be changed to match a directory on your local 
machine containing the data to expose to the docker image. Namely, it must contain folder(s) with images you want to 
run inference on, as long as a folder with the trained models to use, and a destination folder where the results will 
be placed.

For launching the Docker image as a CLI, run:  
```
docker run -v /home/<username>/<resources_path>:/workspace/resources -t -i --network=host --ipc=host --user $(id -u) dbouget/raidionics-segmenter:v1.4-py39-cpu -c /workspace/resources/<path>/<to>/main_config.ini -v <verbose>
```

The `<path>/<to>/main_config.ini` must point to a valid configuration file on your machine, as a relative path to the `/home/<username>/<resources_path>` described above.
For example, if the file is located on my machine under `/home/myuser/Data/Segmentation/main_config.ini`, 
and that `/home/myuser/Data` is the mounted resources partition mounted on the Docker image, the new relative path will be `Segmentation/main_config.ini`.  
The `<verbose>` level can be selected from [debug, info, warning, error].

For running models on the GPU inside the Docker image, run the following CLI, with the gpu_id properly filled in the configuration file:
```
docker run -v /home/<username>/<resources_path>:/workspace/resources -t -i --runtime=nvidia --network=host --ipc=host --user $(id -u) dbouget/raidionics-segmenter:v1.4-py39-cuda12.4 -c /workspace/resources/<path>/<to>/main_config.ini -v <verbose>
```

# [Models](https://github.com/dbouget/raidionics_seg_lib#models)

The trained models are automatically downloaded when running Raidionics or Raidionics-Slicer.  
Alternatively, all existing Raidionics models can be browsed [here](https://github.com/dbouget/Raidionics-models) directly.


# [Developers](https://github.com/dbouget/raidionics_seg_lib#developers)
For running inference on GPU, your machine must be properly configured (cf. [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html))  
In the configuration file, the gpu_id parameter should then point to the GPU that is to be used during inference.

To run the unit tests, type the following within your virtual environment and within the raidionics_seg_lib folder:
```
pip install pytest
pytest tests/
```