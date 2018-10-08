## I have finished the detection branch and am still training the model to verify its correctness. All the features will be published to develop branch, and keep master stable. 
 - ICDAR Dataset 
 - SynthText 800K Dataset
 - detection branch 
 - eval
 - multi-gpu training
 
 
## Questions

- Should I fix weights of the backbone network, resnet50 ?
  ```python
  for param in self.backbone.parameters():
      param.requires_grad = False
  ```
 
 

# Introduction

This is a PyTorch implementation of [FOTS](https://arxiv.org/abs/1801.01671).
