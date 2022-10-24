## Environment Requirement
The required packages are as follows:
* tensorflow == 1.11.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1

Command
```
``python CFAG.py --dataset Mafengwo --regs [1e-2] --embed_size 256 --layer_size [64] --lr 0.001 --aug_type -1 --gat_type both_side --att_ver 5 --att_coef 0.1 --intra_emb_dim 512 --batch_size 2048 --epoch 1000`
````
