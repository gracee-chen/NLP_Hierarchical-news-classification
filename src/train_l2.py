# train_l2.py
import os
import pandas as pd 
from transformers import (AutoModelForSequenceClassification,
                         TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data import (load_level2_df, make_label_maps, encode_dataset, 
                 split_dataset, balance_subcategories, get_consistent_split)
from config import *

def compute_metrics(eval_pred): # same as level 1
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    
    # accuracy
    acc = accuracy_score(labels, preds)
    
    # f1, precision, recall
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_one_parent(parent, sub_df):
    # step1: data preparatin
    # balance subcategories for training
    balanced_df = balance_subcategories(sub_df, parent)
    
    label2id, id2label = make_label_maps(balanced_df.level2)
    ds = encode_dataset(balanced_df, label2id, "level2")
    train_ds, val_ds, test_ds = split_dataset(ds)

    # step2: model initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label2id),
        id2label=id2label, label2id=label2id)

    out_dir = f"{L2_OUT}/{parent}"

    # step3: args
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=L2_LR,
        per_device_train_batch_size=L2_BATCH_SIZE,
        per_device_eval_batch_size=L2_BATCH_SIZE,
        num_train_epochs=L2_EPOCHS,
        eval_strategy="steps",  
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  
        weight_decay=0.01,  
    )

    # step4: train
    trainer = Trainer(
        model=model, 
        args=args,
        train_dataset=train_ds, 
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model(f"{out_dir}/best")
    
    # step5: test
    results = trainer.evaluate(test_ds)
    print(f"{parent} 测试结果: {results}")
    
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/test_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

def main():
    df = load_level2_df()
    
    full_train_df, full_val_df, full_test_df = get_consistent_split(df)
    
    os.makedirs(L2_OUT, exist_ok=True)
    
    # sort parent categories by number of samples, prioritize training categories with more data
    parent_counts = full_train_df['level1'].value_counts()
    sorted_parents = parent_counts.index.tolist()
    
    for parent in sorted_parents:
        train_df = full_train_df[full_train_df['level1'] == parent]
        val_df = full_val_df[full_val_df['level1'] == parent]
        test_df = full_test_df[full_test_df['level1'] == parent]
        
        combined_df = pd.concat([train_df, val_df, test_df])
        train_one_parent(parent, combined_df)

if __name__ == "__main__":
    main()