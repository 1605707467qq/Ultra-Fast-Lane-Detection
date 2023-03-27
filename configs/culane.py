# 数据
dataset='CULane'
data_root = None

# 训练所用到的超参数 epoch batchsize 优化器 学习了 
epoch = 50
batch_size = 32
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9
# 训练策略
scheduler = 'multi' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# 网络结构超参数
use_aux = True
griding_num = 200
backbone = '18'

# 损失函数
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''
# 这个是Tennsorboard的训练日志的保存地址
log_path = None

# 微调或恢复模型 的路径
finetune = None
resume = None

# TEST 测试时 test.py会用的参数
test_model = None
test_work_dir = None

num_lanes = 4




