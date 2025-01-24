## Environment

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

Note: In our environment, Python is called "python3" thats why we use it in the commands. For different users it might be called just "python".

## Data

The preprocessed [ATLAS](https://www.dsimb.inserm.fr/ATLAS/download.html) dataset with topology splits is provided in the folder `data/`. To prepare your own dataset, see following example:

1) Paths for input PDBs and for output directory where to store preprocessed data can be set in `configs/data_config.yaml`.

2) Inside `data/PDBs` place the PDB files of the proteins you want in your dataset. We provide 10 example PDBs from the ATLAS dataset in this repo. The PDB files should be named according to the ATLAS dataset naming convention: PDBCODE_CHAINCODE.pdb (e.g. 1ah7_A.pdb).

3) Run:

``` 
python3 data/scripts/prepare_dataset.py
```

This prepares the `chain_set.jsonl` file, based on the PDB files. Most importantly, it extracts the sequence and the backbone coordinates.

4) Run:

```
python3 data/scripts/get_enm_fluctuations_for_dataset.py --enm ANM
```

This computes the Elastic Network Models (ENM) estimation of per-residue fluctuations for the input dataset. The paths to input file (backbones_dataset_path) and where to output the files with the computed fluctuations are set in `configs/data_config.yaml`. This example uses Anisotropic Network Models (ANM) in particular, but it can also run with Gaussian Network Models (GNM) when specified by the argument.

Alternatively, when specified in the configs, it can also read a .csv file on the input containing paths to PDB files and compute the ENM from there, without the precomputed `chain_set.jsonl` file.


### Reproduction of the dataset of RMSF labels from the ATLAS dataset:

This can take few hours and a significant disc space, as it calls the ATLAS dataset API, downloads the data (including the MD simulations), unzips the data and stores it. It is not necessary to run it for the reproduction as we already provide the preprocessed ATLAS in the repo. If you are building your own dataset, this might be irrelevant, unless your proteins of interest are included in the ATLAS dataset.

To download ATLAS dataset (in order to obtain the RMSF labels for the training), run the following command:

```
python3 data/atlas/download_analyses.py
```

To extract the RMSF labels from the ATLAS dataset run:

```
python3 data/scripts/extract_rmsf_labels.py
```

Paths for input / output for the RMSF label extraction can be modified in `configs/data_config.yml`.

If you use the ATLAS dataset, please cite the [paper](https://academic.oup.com/nar/article/52/D1/D384/7438909?login=false) by Meersche et al.

## Training Flexpert-Seq and Flexpert-3D

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

The code for the LoRA fine-tuning of protein language models is derived from [this repo](https://github.com/agemagician/ProtTrans/tree/master/Fine-Tuning) accompanying the [paper](https://www.nature.com/articles/s41467-024-51844-2) "Fine-tuning protein language models boosts predictions across diverse tasks" by Schmirler et al.

## Inference with Flexpert-Seq and Flexpert-3D
Example predictions of flexibility:
```
#For Flexpert-Seq:
python3 predict.py --modality SEQ --input_file data/example_sequences.fasta 

#For Flexpert-3D:
python3 predict.py --modality 3D --input_file data/custom_dataset/chain_set.jsonl
```

## Analysis of the flexibility metrics

TODO