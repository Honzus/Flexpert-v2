# Flexpert-Design

## Training

First make sure you have the Flexpert-3D model weights in the `../models/weights` directory. Alternatively run the following script to download the weights.

```bash
. ../download_flexpert_weights.sh
```

Then run the following command to train the model.

```bash
export HF_HOME=./HF_cache
python3 train.py \
    --batch_size 4 \
    --dataset FLEX_CATH4.3 \
    --ex_name train_repro \
    --offline 0 \
    --gpus 1 \
    --epoch 20 \
    --flex_loss_coeff 0.9 \
    --use_pmpnn_checkpoint 1
```

## Inference

#TODO:
- describe the predict.py usage

TODO:
- host weights and data and provide the scripts to download them
- make predict.py work
    - todo make downloadable weights
    - check that init_flex_features takes effect
    - todo prepare the reading of the flexibility instructions from a separate file

- move to public repo
- delete obviously unnecessary files and code