import torch

#测试torch环境和cuda是否可用
print(torch.__version__)
print(torch.cuda.is_available())
