# data.py
import os, pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.utils import resample
from config import *

# initialize BERT tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_csv():
    df = pd.read_csv(CSV_PATH, usecols=[
        "content", "category_level_1", "category_level_2"
    ]).dropna() # drop row with missing value
    return df.rename(columns={"content": "text",
                              "category_level_1": "level1",
                              "category_level_2": "level2"})

def make_label_maps(series): # map label with specific ID
    classes = sorted(series.unique())
    return {c:i for i,c in enumerate(classes)}, {i:c for i,c in enumerate(classes)}

def encode_dataset(df, label2id, label_col):
    def _encode(batch):
        enc = tok(batch["text"],
                  truncation=True,
                  padding="max_length",
                  max_length=MAX_LEN)
        labels = [label2id[str(lbl)] for lbl in batch[label_col]]
        enc["labels"] = labels
        return enc

    return Dataset.from_pandas(df).map(
        _encode, batched=True, remove_columns=list(df.columns)
    )

def split_dataset(ds):
    # 80% trained data, 10% validation data, 10% test data
    train_test = ds.train_test_split(test_size=0.2, seed=RANDOM_SEED)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=RANDOM_SEED)
    return train_test["train"], val_test["train"], val_test["test"]

def balance_subcategories(df, parent_category):
    """sampling subcategories"""
    subcat_df = df[df['level1'] == parent_category]
    
    subcats = subcat_df['level2'].value_counts()
    
    target_size = max(int(subcats.median()), 50)
    
    balanced_dfs = []
    for subcat, count in subcats.items():
        subcat_samples = subcat_df[subcat_df['level2'] == subcat]
        
        if count < target_size:
            upsampled = resample(subcat_samples, 
                                replace=True,
                                n_samples=target_size,
                                random_state=RANDOM_SEED)
            balanced_dfs.append(upsampled)
        else:
            balanced_dfs.append(subcat_samples)
    
    return pd.concat(balanced_dfs)

# ---------- first level ----------
def load_level1_df():
    return load_csv()[["text", "level1"]]

# ---------- second level ----------
def load_level2_df():
    return load_csv()[["text", "level1", "level2"]]

def get_consistent_split(df):
    df['id'] = range(len(df))
    
    ids = np.array(df['id'])
    from sklearn.model_selection import train_test_split
    
    # first split：80%train，20%valid+test
    train_ids, test_val_ids = train_test_split(
        ids, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # second split：50%test，50%validation
    val_ids, test_ids = train_test_split(
        test_val_ids, test_size=0.5, random_state=RANDOM_SEED
    )
    
    train_mask = df['id'].isin(train_ids)
    val_mask = df['id'].isin(val_ids)
    test_mask = df['id'].isin(test_ids)
    
    return df[train_mask], df[val_mask], df[test_mask]