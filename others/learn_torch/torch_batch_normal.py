import torch
import torch.nn as nn

batch_size = 2
time_steps = 3
embedding_dim = 4

# NLP中
inputx = torch.randn(batch_size, time_steps, embedding_dim) #N*C*L

# 1. 实现batch_normal 并验证
'''per channel across mini-batch'''
batch_norm_op = torch.nn.BatchNorm1d(embedding_dim,affine = False)
bn_y = batch_norm_op(inputx.transpose(-1,-2)).transpose(-1,-2)
## 上面的是官方给的

'''手写batch-normal'''
'''均值部分有两个部分都要求均值，1 batch ；2channel'''
bn_mean= inputx.mean(dim = (0,1),keepdim=True)
bn_std = inputx.std(dim = (0,1),keepdim=True,unbiased=False) # unbiased (bool) - 是否使用基于修正贝塞尔函数的无偏估计
verify_bn_y = (inputx-bn_mean)/(bn_std+1e-5)
# print(bn_y)
# print(verify_bn_y)

# 2. 调用layer normal
'''per sample per layer 只需要实例化输出是最后一维，也就是说只对最后一个维度计算均值 '''
layer_norm_op = torch.nn.LayerNorm(embedding_dim,elementwise_affine = False)
ln_y = layer_norm_op(inputx)
'''手写layer normal'''
ln_mean = inputx.mean(dim = -1,keepdim=True)
ln_std = inputx.std(dim = -1,keepdim=True,unbiased=False)
verify_ln_y = (inputx - ln_mean)/(ln_std+1e-5)
print(ln_y)
print(verify_ln_y)
print(ln_y==verify_ln_y)