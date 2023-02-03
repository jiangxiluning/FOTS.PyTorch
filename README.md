# Introduction

This is a PyTorch implementation of [FOTS](https://arxiv.org/abs/1801.01671).

 - [x] ICDAR Dataset 
 - [x] SynthText 800K Dataset
 - [x] detection branch 
 - [x] recognition branch
 - [x] eval
 - [x] multi-gpu training
 - [x] reasonable project structure
 - [x] wandb
 - [x] pytorch_lightning
 - [x] eval with different scales

# Instruction

## Requirements

1. build tools

   ```
   ./build.sh
   ```

2. prepare Dataset

3. create virtual env, you may need conda
   ```
   conda create --name fots --file spec-file.txt
   conda activate fots
   pip install -r reqs.txt
   ```


## Training

   ```
   # quite easy, for single gpu training set gpus to [0]. 0 is the id of your gpu.
   python train.py -c pretrain.json
   python train.py -c finetune.json

   ```
   
## Evaluation

```
python eval.py 
-c
finetune.json
-m
<your ckpt>
-i
<icdar2015 folder contains train and test>
--detection    
-o
./results
--cuda
```

with `--detection` flag to evaluate detection only or without flag to evaluate e2e

## Benchmarking and Models (Coming soon!)

### Visualization (1000 epochs, 8 bs, icdar2015 without finetuning, still converging!!!)

![img_59.jpg](https://s2.loli.net/2022/05/04/entWAbuEoYNV6sP.jpg)
![img_108.jpg](https://s2.loli.net/2022/05/04/B4Qdg2C6ZcbF89q.jpg)


## Acknowledgement
- https://github.com/SakuraRiven/EAST (Some codes are copied from here.)
- https://github.com/chenjun2hao/FOTS.pytorch.git (ROIRotate)
