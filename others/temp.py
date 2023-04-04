import torch
import time
import numpy as np
import torchvision
import torch.nn.modules
# from efficientnet_pytorch import EfficientNet
import csv

filePath = 'temp.csv'
max = ['MobileNetV2','mobilenet_v2','mobilenetv3','mobilenet_v3_small','efficientnet_b0','vgg11_bn','efficientnet_b7']

pretrained = False
bad = []
good = ['resnet50','mnasnet1_3','vgg11','vgg13','AlexNet','','shufflenet_v2_x1_0','resnet18','mobilenet_v2','vgg19','vgg13_bn','vgg11_bn','vgg16','squeezenet1_1','mnasnet0_5','mobilenet_v3_large']
model_result = []
nets = ['AlexNet', 'DenseNet', 'EfficientNet', 'GoogLeNet', 'GoogLeNetOutputs', 'Inception3', 
'InceptionOutputs', 'MNASNet', 'MobileNetV2', 'MobileNetV3', 'RegNet', 'ResNet', 'ShuffleNetV2',
 'SqueezeNet', 'VGG' 'alexnet', 'densenet', 'densenet121',
  'densenet161', 'densenet169', 'densenet201', 'detection', 
  'efficientnet', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 
  'feature_extraction', 'googlenet', 'inception', 'inception_v3', 'mnasnet', 'mnasnet0_5', 
  'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet', 'mobilenet_v2', 'mobilenet_v3_large', 
  'mobilenet_v3_small', 'mobilenetv2', 'mobilenetv3', 'quantization', 'regnet', 'regnet_x_16gf',   'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 
    'regnet_x_8gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf',   'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf',
     'resnet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 
     'segmentation', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenetv2',
      'squeezenet', 'squeezenet1_0', 'squeezenet1_1', 'vgg', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'video', 'wide_resnet101_2', 'wide_resnet50_2']
img = torch.zeros((1,3,288,800)).cuda() + 1
# ... image preprocessing as in the classification example ...
print(img.shape) # torch.Size([1, 3, 224, 224])


for net in nets:
    try:
        k = getattr(torchvision.models,net)
        model= k().cuda()
        t_all = []
        for i in range(200):
            t1 = time.time()
            y = model(img)
            t2 = time.time()
            t_all.append(t2 - t1)
        t_all = [i for i in t_all if i != 0]
        print(net+'average time:', np.mean(t_all) / 1)
        print(net+'average fps:',1 / np.mean(t_all))

        print(net+'fastest time:', min(t_all) / 1)
        print(net+'fastest fps:',1 / min(t_all))

        # print('slowest time:', max(t_all) / 1)
        # print('slowest fps:',1 / max(t_all))
        model_result = [net,np.mean(t_all) / 1,1 / np.mean(t_all),min(t_all) / 1,1 / min(t_all)]#, max(t_all) / 1,1 / max(t_all)]

        rows = [i for i in model_result]

        with open(filePath,'a+') as f:
            csv_write = csv.writer(f)
            data_row = rows
            csv_write.writerow(data_row)
    except:continue
