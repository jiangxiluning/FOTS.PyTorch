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

## Benchmarking and Models 
Belows are **E2E Generic** benchmarking results on the ICDAR2015. I pretrained on Synthtext (7 epochs).  [Pretrained model](https://pan.baidu.com/s/18RR9J7TvuZn4LUCv2eJmHQ) (code: 68ta). Finetuned model will be released soon.



| Name            | Backbone  | Scale (W * H) | Heamn |
|-----------------|-----------|---------------|-------|
| FOTS (paper)    | Resnet 50 | 2240 * 1260   | 60.8  |
| FOTS (ours)     | Resnet 50 | 2240 * 1260   | TBR   |
| FOTS RT (paper) | Resnet 34 | 1280 * 720    | 51.4  |
| FOTS RT (Ours) | Resnet 50 | 1280 * 720    | TBR   |



## Acknowledgement
- https://github.com/SakuraRiven/EAST (Some codes are copied from here.)
- https://github.com/chenjun2hao/FOTS.pytorch.git (ROIRotate)
