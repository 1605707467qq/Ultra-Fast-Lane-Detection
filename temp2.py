import torch
import time
import numpy as np
img = torch.zeros((1,3,294,840)).cuda() + 1
model= torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()

t_all = []
for i in range(200):
    t1 = time.time()
    y = model(img)
    t2 = time.time()
    t_all.append(t2 - t1)
t_all = [i for i in t_all if i != 0]
print(model+'average time:', np.mean(t_all) / 1)
print(model+'average fps:',1 / np.mean(t_all))

print(model+'fastest time:', min(t_all) / 1)
print(model+'fastest fps:',1 / min(t_all))

# print('slowest time:', max(t_all) / 1)
# print('slowest fps:',1 / max(t_all))
model_result = [model,np.mean(t_all) / 1,1 / np.mean(t_all),min(t_all) / 1,1 / min(t_all)]#, max(t_all) / 1,1 / max(t_all)]
print(model_result)
