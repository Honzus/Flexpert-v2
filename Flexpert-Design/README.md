# Flexpert-Design

## Training

First make sure you have the Flexpert-3D model weights in the `../models/weights` directory. Alternatively run the following script to download the weights.

```bash
. ../download_flexpert_weights.sh
```

Then run the following command to train the model.

```bash
export HF_HOME=./HF_cache
python3 main.py \
    --batch_size 4 \
    --model_name 'ProteinMPNN' \
    --stage 'fit' \
    --dataset FLEX_CATH4.3 \
    --ex_name train_repro \
    --offline 0 \
    --gpus 1 \
    --epoch 20 \
    --use_dynamics 1 \
    --flex_loss_coeff 0.9 \
    --init_flex_features 1 \
    --grad_normalization 0 \
    --loss_fn MSE \
    --use_pmpnn_checkpoint 1
```

## Inference

#TODO