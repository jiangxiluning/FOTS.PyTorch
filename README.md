## I have finished the detection branch and am still training the model to verify its correctness. All the features will be published to develop branch, and keep master stable. 
 - ICDAR Dataset 
 - SynthText 800K Dataset
 - detection branch (verified on the training set, It works!)
 - eval
 - multi-gpu training
 
 
## Questions

- Should I fix weights of the backbone network, resnet50 ?
  ```python
  for param in self.backbone.parameters():
      param.requires_grad = False
  ```
  Answer: Yes, the backbone network is used as a feature extractor, so we do not need to modify the weights.
 
 

# Introduction

This is a PyTorch implementation of [FOTS](https://arxiv.org/abs/1801.01671).
