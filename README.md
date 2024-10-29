### Environment

The `environment.txt` file can be used to create your Python environment.

Alternatively, use the Docker (Singularity) images with PyTorch, PyTorch Geometric and Pytorch Lightning built for (i) NVIDIA GPUs (CUDA), (ii) AMD GPUs (ROCm), see example below:

```
TODO: instructions for singularity
```

Some packages might still be missing, but the crucial packages depending on the GPU drivers should work properly. The missing packages can be installed with pip.

We will provide a complete image in the future.

### Data

TODO

### Analysis of the flexibility metrics

TODO


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
