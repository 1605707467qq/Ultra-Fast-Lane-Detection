# 数据集 名称 位置
dataset='Tusimple'
data_root = 'D:/Programme/dataset/Tusimple/Tusimple'

# 训练参数
epoch = 100 
batch_size = 2
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# 网络参数
backbone = '18'
# 主干网络 # 每一行分类的个数
griding_num = 100
use_aux = False

# 损失函数
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''
# 这个是Tennsorboard的训练日志的保存地址
log_path = 'train_log'

# 微调或恢复模型 的路径
finetune = None
resume = None

# TEST test.py会用的参数
test_model = 'D:/Programme/Ultra-Fast-Lane-Detection/tusimple_18.pth'
test_work_dir = 'D:/Programme/dataset/Tusimple/Tusimple'

num_lanes = 4