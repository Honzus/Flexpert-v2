### Environment

The `environment.txt` file can be used to create your Python environment.

Alternatively, use the Docker (Singularity) images with PyTorch, PyTorch Geometric and Pytorch Lightning built for (i) NVIDIA GPUs (CUDA), (ii) AMD GPUs (ROCm), see example below:

```
#Set environment variables for singularity cache, it should be on a disc with enough free space (tens of GB) - the provided path is an example which works well for our cluster
export SINGULARITY_TMPDIR=/tmp/USERNAME
export SINGULARITY_CACHEDIR=/tmp/USERNAME

#For AMD GPUs pull this image:
singularity pull docker://koubic/lumi-pyg-lightning-tk:latest

#For GPUs with CUDA support pull this:
singularity pull docker://koubic/karolina_cuda_pyg:latest

#On the GPU node (e.g. after allocating interactive job on a GPU node), activate the singularity container e.g. like this (mounting the /scratch drive, mount the directory relevant for you):
singularity exec -B /scratch/:/scratch/ lumi-pyg-lightning-tk_latest.sif bash #Or use the other container in case of CUDA machine

```

Some packages might still be missing, but the crucial packages depending on the GPU drivers should work properly. The missing packages can be installed with pip.

We will provide a complete image in the future.

### Data

The preprocessed ATLAS dataset with topology splits is provided in the folder `data/`. To prepare your own dataset, see following example:

TODO: start with "RMSF predictor training" notes

``` 

```

### Training Flexpert-Seq and Flexpert-3D

Inside `config/` review the 3 config files: 

1) `lora_config.yaml` contains the default LoRA parameters, from this repo (and corresponding paper). Leave this as it is unless you want to make your own experiments.
2) `train_config.yaml` contains arguments to reproduce the training. It can be changed to experiment, alternatively most of these arguments can be overriden by arguments passed to the `train.py` script. See `python3 train.py --help` for the arguments which can be provided directly to the script.
3) `env_config.yaml` use this to set cache path for HuggingFace models or to set name of wandb project.

Run the training:
```
#For training Flexpert-Seq:
python3 train.py --run_name testrun-Seq --adaptor_architecture no-adaptor

#For training Flexpert-3D:
python3 train.py --run_name testrun-3D --adaptor_architecture conv
```

### Inference with Flexpert-Seq and Flexpert-3D

TODO


### Analysis of the flexibility metrics

TODO