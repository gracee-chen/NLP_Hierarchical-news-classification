# config.py
CSV_PATH   = "./data/processed/filtered_level.csv"
MODEL_NAME = "bert-base-uncased" #pretrained-model we used
MAX_LEN    = 256

# first level parameter
L1_BATCH_SIZE = 16
L1_EPOCHS     = 4
L1_LR         = 2e-5

# second level parameter
L2_BATCH_SIZE = 8
L2_EPOCHS     = 8
L2_LR         = 5e-5

RANDOM_SEED = 42

L1_OUT = "outputs/l1"
L2_OUT = "outputs/l2"