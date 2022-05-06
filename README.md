# Raidionics backend for segmentation/classification

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Build Actions Status](https://github.com/dbouget/raidionics-seg-lib/workflows/Build/badge.svg)](https://github.com/dbouget/raidionics-seg-lib/actions)
[![Paper](https://zenodo.org/badge/DOI/10.1038/s41598-017-17204-5.svg)](https://doi.org/10.48550/arXiv.2204.14199)

The code corresponds to the TensorFlow segmentation or classification backend of MRI/CT volumes.  
The module can either be used as a Python library, as CLI, or as Docker container.

## 1. Trained segmentation models
The models and associated configuration files to be used should be placed inside a folder
at the project root directory. The folder name should be: *resources/models/*, and there
should be one sub-folder for each separate models (e.g., MRI_Tumor, CT_LymphNodes).  
The following is an example of the folder structure:
> * ./  
> --- resources/  
> ------ models/  
> --------- MRI_Brain/  
> ------------ model.hd5  
> ------------ pre_processing.ini

## 2. Runtime segmentation configuration
A runtime configuration file (runtime_config.ini) can be used to specify some user choices.
The configuration file should look like this:  

>[Predictions]  
>non_overlapping=False  
>reconstruction_method= #argmax, probabilities, thresholding  
>reconstruction_order= #resample_first, resample_second  
>probability_threshold=0.5  

The configuration file should be placed within */resources/data*:
> * ./  
> --- resources/  
> ------ data/  
> --------- runtime_config.ini  

## 3. Command line execution
The usage is:  
> main.py --Task *TaskName* --Input *InputFilename* --Output *OutputDirname* --Model *model_name* --GPU *gpu_id*

* Task: segmentation or classification.  
* Input: complete filename of the input MRI volume to process.  
* Output: folder where to dump the computed results.  
* Model: name of the sub-folder containing the model to be used (e.g., MRI_Brain).  
* GPU: id of the GPU to use, -1 to use the CPU (default).  

## 4. Creating the Docker image
The Dockerfile specifies the content of the Docker image, and the following lines should be 
executed to update the image with the latest source code modifications. The created image is used 
as backend for the 3D Slicer plugin.  

> docker build --network=host -t *username*/*image\_name*:*tag* */path/to/Dockerfile*  
> docker push *username*/*image\_name*:*tag*  
