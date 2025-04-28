# train_l1.py
import sys, pathlib
import transformers
from evaluate import load as load_metric
from transformers import (AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data import load_level1_df, make_label_maps, encode_dataset, split_dataset
from config import *

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
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

def main():
    print(f"transformers version: {transformers.__version__}")
    
    # step1: data preparation
    df = load_level1_df()
    label2id, id2label = make_label_maps(df["level1"])
    ds = encode_dataset(df, label2id, "level1")
    train_ds, val_ds, test_ds = split_dataset(ds)

    # step2: model initialization (label corresponding to level1)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label2id), id2label=id2label,
        label2id=label2id)

    # step3: agrs
    args = TrainingArguments(
        output_dir=L1_OUT,
        learning_rate=L1_LR,
        per_device_train_batch_size=L1_BATCH_SIZE,
        per_device_eval_batch_size=L1_BATCH_SIZE,
        num_train_epochs=L1_EPOCHS,
        logging_steps=50,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",  
        save_strategy="steps",  
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=0.01,
    )

    # step4: train
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(f"{L1_OUT}/best")
    
    # test
    results = trainer.evaluate(test_ds)
    print(f"测试结果: {results}")
    
    import os
    os.makedirs(L1_OUT, exist_ok=True)
    with open(f"{L1_OUT}/test_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()