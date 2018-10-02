## 项目还未完全完成，后续的开发会采用 git flow 版本模型， 开发都在 dev 上做， 稳定特性在 master。 目前 master 分支支持的 feature 有：
 - ICDAR 数据集读取
 - SynthText 800K 数据集读取
 - detection branch 的 loss 计算
 - eval 支持验证模型
 - 多 gpu 训练
 
 
## Questions

- Should I fix weights of the backbone network, resnet50 ?
  ```python
  for param in self.backbone.parameters():
      param.requires_grad = False
  ```
 
 

# Introduction

This is a PyTorch implementation of [FOTS](https://arxiv.org/abs/1801.01671).
