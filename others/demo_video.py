"""
2021.08.16
author:alian
function: 
Ultra-Fast-Lane-Detection 测试视频文件并保存结果为视频
"""

import torch, os
import cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cuda()  # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        # transforms.CenterCrop((590,1640)),
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    cap = cv2.VideoCapture("******.avi")  # 读取视频文件
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    rval, frame = cap.read()
    frame = frame[490:1080, 0:1640, :]
    vout = cv2.VideoWriter('output.avi', fourcc, 30.0, (frame.shape[1], frame.shape[0]))  # 结果保存
    print("w= {},h = {}".format(cap.get(3), cap.get(4)))
    from PIL import Image

    print('加载CUDA是否成功:', torch.cuda.is_available())
    while 1:
        rval, frame = cap.read()
        if rval == False:
            break
        frame = frame[490:1080, 0:1640, :]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_ = Image.fromarray(img)
        imgs = img_transforms(img_)
        imgs = imgs.unsqueeze(0)
        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc

        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (
                        int(out_j[k, i] * col_sample_w * frame.shape[1] / 800) - 1, int(frame.shape[0] - k * 20) - 1)
                        cv2.circle(frame, ppp, 5, (0, 255, 0), -1)
        vout.write(frame)

    vout.release()

