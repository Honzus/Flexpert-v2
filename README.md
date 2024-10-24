To run training:

```
python3 train.py --data_path data/rmsf_atlas_data_prottransready.txt --run_name debug --batch_size 4 --epochs 2 --save_steps 528 --fasta_path data/atlas_sequences.fasta --enm_path data/atlas_minimized_fluctuations_ANM.jsonl --splits_path data/atlas_splits.json --adaptor_architecture no-adaptor
```