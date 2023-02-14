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
 - [ ] OHEM

# Instruction

## Requirements
   ```
   conda create --name fots --file spec-file.txt
   conda activate fots
   pip install -r reqs.txt

   cd FOTS/rroi_align
   python build.py develop
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
-c finetune.json
-m <your ckpt>
-i <icdar2015 folder contains train and test>
--detection    
-o ./results
--cuda
--size "1280 720"
--bs 2
--gpu 1
```

with `--detection` flag to evaluate detection only or without flag to evaluate e2e

## Benchmarking and Models 
Belows are **E2E Generic** benchmarking results on the ICDAR2015. I pretrained on Synthtext (7 epochs).  [Pretrained model](https://pan.baidu.com/s/18RR9J7TvuZn4LUCv2eJmHQ) (code: 68ta). [Finetuned (5000 epochs) model](https://pan.baidu.com/s/14UnlBP5xfRXx90bdlIBAEg) (code: s38c).



| Name            | Backbone  | Scale (W * H) | Hmean |
|-----------------|-----------|---------------|-------|
| FOTS (paper)    | Resnet 50 | 2240 * 1260   | 60.8  |
| FOTS (ours)     | Resnet 50 | 2240 * 1260   | 46.2 |
| FOTS RT (paper) | Resnet 34 | 1280 * 720    | 51.4  |
| FOTS RT (Ours) | Resnet 50 | 1280 * 720    | 47   |

## Samples

![img_295.jpg](https://s2.loli.net/2023/02/14/PiJWMIoFvGtsqu8.jpg)
![img_486.jpg](https://s2.loli.net/2023/02/14/3p8PeyqFCUYtOvg.jpg)
![img_497.jpg](https://s2.loli.net/2023/02/14/mODBYHzr7gle6Qq.jpg)



## Acknowledgement
- https://github.com/SakuraRiven/EAST (Some codes are copied from here.)
- https://github.com/chenjun2hao/FOTS.pytorch.git (ROIRotate)
