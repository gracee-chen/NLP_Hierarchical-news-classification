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
    
    # 计算准确率
    acc = accuracy_score(labels, preds)
    
    # 计算精确率、召回率和F1值
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
    # 打印当前的transformers版本，帮助调试
    print(f"当前transformers版本: {transformers.__version__}")
    
    df = load_level1_df()
    label2id, id2label = make_label_maps(df["level1"])
    ds = encode_dataset(df, label2id, "level1")
    train_ds, val_ds, test_ds = split_dataset(ds)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label2id), id2label=id2label,
        label2id=label2id)

    # 根据版本不同调整参数
    args = TrainingArguments(
        output_dir=L1_OUT,
        learning_rate=L1_LR,
        per_device_train_batch_size=L1_BATCH_SIZE,
        per_device_eval_batch_size=L1_BATCH_SIZE,
        num_train_epochs=L1_EPOCHS,
        # 移除可能不兼容的参数
        logging_steps=50,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",  # 适用于旧版本的参数名称
        save_strategy="steps",  # 适用于旧版本的参数名称
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=0.01,  # 添加权重衰减，减少过拟合
    )

    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(f"{L1_OUT}/best")
    
    # 在测试集上评估
    results = trainer.evaluate(test_ds)
    print(f"测试结果: {results}")
    
    # 保存评估结果
    import os
    os.makedirs(L1_OUT, exist_ok=True)
    with open(f"{L1_OUT}/test_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()