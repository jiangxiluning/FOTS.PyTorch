# News!!! Recognition branch now is added into model. The whole project has beed optimized and refactored.
 - [x] ICDAR Dataset 
 - [x] SynthText 800K Dataset
 - [x] detection branch (verified on the training set, It works!)
 - [x] recognition branch
 - [ ] eval
 - [x] multi-gpu training
 - [x] reasonable project structure
 - [x] wandb
 - [x] pytorch_lightning

 

# Introduction

This is a PyTorch implementation of [FOTS](https://arxiv.org/abs/1801.01671).

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
python eval.py -m <model.tar.gz> -i <input_images_folder> -o <output_folders>

```

## Benchmarking and Models (Coming soon!)


## Acknowledgement
- https://github.com/SakuraRiven/EAST (Some codes are copied from here.)
- https://github.com/chenjun2hao/FOTS.pytorch.git (ROIRotate)
