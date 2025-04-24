# config.py
CSV_PATH   = "filtered_level1.csv"
MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 256

# 分离一层和二层的训练参数
L1_BATCH_SIZE = 16
L1_EPOCHS     = 4
L1_LR         = 2e-5

# 二层使用更高的学习率和更多训练轮数
L2_BATCH_SIZE = 8
L2_EPOCHS     = 8
L2_LR         = 5e-5

RANDOM_SEED = 42

L1_OUT = "outputs/l1"
L2_OUT = "outputs/l2"