import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos

'''使用PIL库打开图片，不仅可以打开jpg 还可以打开png'''
def loader_func(path):
    return Image.open(path)

'''导入车道线的测试集，待看，先看训练集的dataset吧'''
class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane


    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)

'''导入车道线的训练集'''
class LaneClsDataset(torch.utils.data.Dataset):
    '''初始化的时候需要给我 路径，列表路径，图像增强的方式，标注的增强的变换，              光照数据增强            栅格数量        车道的名称
                行锚                是否使用辅助分支   语义分割的变换       车道线的最大数量          '''
    def __init__(self, path, list_path, img_transform = None,target_transform = None,simu_transform = None, griding_num=50, load_name = False,
                row_anchor = None,use_aux=False,segment_transform=None, num_lanes = 4):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform# simu_transform是Ultra Fast lane detection算法中用于数据增强的一个函数1。它可以对输入图像进行随机的仿射变换，以模拟不同的视角和光照条件2。
        self.path = path# '/home/mengxc/dataset/Tusimple'
        self.griding_num = griding_num
        self.load_name = load_name# Land_name是Ultra Fast lane detection算法中用于表示车道线的标签1
        self.use_aux = use_aux
        self.num_lanes = num_lanes
        # list_path'/home/mengxc/dataset/Tusimple/train_gt.txt'
        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor # anchor的所在位置
        self.row_anchor.sort()
    '''一般会先走__getitem__函数，因为这里定义里如何去处理数据的，索引是随机抽取第几个图片和标注'''
    def __getitem__(self, index):
        l = self.list[index]#index=269
        l_info = l.split()# 'list的第269行是clips/0601/1494452553518276564/20.jpg clips/0601/1494452553518276564/20.png 1 1 1 1\n'
        img_name, label_name = l_info[0], l_info[1]# 划分一下得到数据和标注图片的相对路径
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]
        '''上面是读取数据和标签的路径的，下面是通过图片和标注的相对路径读取车道线'''
        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)
    
        '''这里代表的是数据增强，其实还是蛮繁琐的，判断了grid 并且拟合了车道线'''
        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
        lane_pts = self._get_index(label)

        '''获取行锚处的车道坐标,并且会使用一阶线性拟合延申车道线到行锚的位置末端'''
        '''获取到标签值还没完还要得到网格的值——_gridp_ts
        得到所有标签值还没完，还要得到网格的一个值'''
        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        '''到这儿标签就全部处理好了，这就是一会去算损失函数的时候会用到的标签             制作坐标分类标签'''
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)
        '''数据增强'''
        if self.img_transform is not None:
            img = self.img_transform(img)
        '''增强预训练模型backbone提取特征的能力'''
        if self.use_aux:
            return img, cls_label, seg_label
        if self.load_name:
            return img, cls_label, img_name
        return img, cls_label
        '''废了好大劲儿，终于把标签弄完了'''
    def __len__(self):
        return len(self.list)
    '''得到所有标签值还没完，还要得到网格的一个值 pts.shape:4，56，2 | w=1280'''
    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2 
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)# 等差数列 np.linspace(0, w - 1, num_cols)是一个NumPy函数，它的作用是生成一个等差数列，即将区间[0, w-1]平均分成num_cols份，并返回一个包含这些分割点的NumPy数组。
        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
            '''就是把它映射到一个由等差数列所产生的格子中。如果pt==-1 设为100，如果pt！=-1就设置为pt//等差数列的长度 这段代码的含义是将pt除以(col_sample[1] - col_sample[0])的商取整并转换为整数，即将pt映射到一个以(col_sample[1] - col_sample[0])为宽度的像素格子中，返回这个像素格子的索引。'''
        return to_pts.astype(int)
    '''获取行锚处的车道坐标，所输入的label是包含车道线语义分割的PNG图片'''
    def _get_index(self, label):
        w, h = label.size
        '''原始是720*1280对标签进行一个缩放的操作，resized to 288×800'''
        if h != 288:
            scale_f = lambda x : int((x * 1.0/288) * h)
            sample_tmp = list(map(scale_f,self.row_anchor))# 先验值：row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, ...]
            #[160, 170, 180, 190, 200, 210, 220, 229, 240, 250, 260, 270, 280, 290, ...710]
        all_idx = np.zeros((self.num_lanes,len(sample_tmp),2)) # 4*56*2 车道线的数量，先验值的数量，2：第一个位置0表示的是这个索引指的是第几行第二个1表示的是代表的是的-1或者车道线的位置。
        for i,r in enumerate(sample_tmp): # [160, 170, 180, 190, 200, 210, 220, 229, 240, 250, 260, 270, 280, 290, ...,710]
            label_r = np.asarray(label)[int(round(r))] # round(160) 是 Python 内置函数 round() 的调用，它的作用是将给定的数字四舍五入为最接近的整数。
            for lane_idx in range(1, self.num_lanes + 1):# 第1条到第n条车道线 lane_idx代表第几条车道线
                pos = np.where(label_r == lane_idx)[0] # 定位等于1的车道线的位置
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r # 第n条车道线，第i个位置索引，第一个代表是第几行
                    all_idx[lane_idx - 1, i, 1] = -1 # 第n条车道线，第i个位置索引，第一个代表是这一行是否有车道线，－1代表没有。pos代表车道线所在的位置
                    continue
                pos = np.mean(pos) # array([725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737,738, 739, 740, 741]) --》733.0
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos # 第n条车道线，第i个位置索引，第一个代表是这一行是否有车道线，pos代表车道线所在的位置
        '''i代表56个索引的其中一项，r代表的是该索引对应的是图片的第几行'''

        '''数据增强：将车道延伸到图像的边界           下面是这段代码的用途'''
        all_idx_cp = all_idx.copy()# 4*56*2
        for i in range(self.num_lanes):
            '''判断车道线是否需要拟合'''
            if np.all(all_idx_cp[i,:,1] == -1):
                continue
            # 如果没有车道
            valid = all_idx_cp[i,:,1] != -1 # 4*56*2
            # 获得所有有效车道点的索引
            valid_idx = all_idx_cp[i,valid,:] # array([False, False, False, False, False, False, False, False, False, True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True,True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True])
            # 获得所有有效的车道点
            if valid_idx[-1,0] == all_idx_cp[0,-1,0]:
                # 如果最后一个有效的车道点的y坐标已经是所有行的最后一个y坐标的话
                # 这意味着这个车道已经到达了图像的底部边界。
                # 所以我们跳过
                continue
            if len(valid_idx) < 6:
                continue
            # 如果车道太短，也就是valid_idx的长度连6都没有就，无法延长
            '''开始拟合车道线，使用一阶线性拟合。这里其实是可以修改为二阶三阶的或者贝塞尔曲线的'''
            valid_idx_half = valid_idx[len(valid_idx) // 2:,:]  # 使用后半部分的数据进行拟合 38*2 ->
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1) # 获取这个‘1阶’线性拟合后得到的线的参数
            start_line = valid_idx_half[-1,0] # 设定拟合的初始位置是val index的长度的一半的位置所对应的索引 这里是630.0
            pos = find_start_pos(all_idx_cp[i,:,0],start_line) + 1  # from data.mytransforms import find_start_pos 
            '''上面的是630，就是要找一个all_idx_half中索引值最为相似的，也就是630.0放在哪里合适，这里是47的索引，所以说要把近似值变为准确的行级分类器的索引值'''
            fitted = np.polyval(p,all_idx_cp[i,pos:,0])# pos=48 第i条车道线从48开始
            fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])
            '''得到的值是：array([-1, -1, -1, -1, -1, -1, -1, -1]) 其实是不太理想的，确实是需要改进一下'''
            assert np.all(all_idx_cp[i,pos:,1] == -1)
            all_idx_cp[i,pos:,1] = fitted # 这里全是-1 代表没有点
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace() # 在Python中，pdb是一个内置的调试器，可以用来在代码执行过程中暂停程序并查看当前的状态，从而进行调试和错误修复。
        return all_idx_cp
