# train_l2.py
import os
from transformers import (AutoModelForSequenceClassification,
                         TrainingArguments, Trainer, EarlyStoppingCallback)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data import (load_level2_df, make_label_maps, encode_dataset, 
                 split_dataset, balance_subcategories, get_consistent_split)
from config import *

def compute_metrics(eval_pred):
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

def train_one_parent(parent, sub_df):
    # 应用数据平衡策略
    balanced_df = balance_subcategories(sub_df, parent)
    
    label2id, id2label = make_label_maps(balanced_df.level2)
    ds = encode_dataset(balanced_df, label2id, "level2")
    train_ds, val_ds, test_ds = split_dataset(ds)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label2id),
        id2label=id2label, label2id=label2id)

    out_dir = f"{L2_OUT}/{parent}"
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=L2_LR,  # 使用更高的学习率
        per_device_train_batch_size=L2_BATCH_SIZE,
        per_device_eval_batch_size=L2_BATCH_SIZE,
        num_train_epochs=L2_EPOCHS,  # 增加训练轮数
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # 使用F1分数而非loss
        weight_decay=0.01,  # 添加权重衰减
        warmup_ratio=0.1,   # 添加学习率预热
    )

    trainer = Trainer(
        model=model, 
        args=args,
        train_dataset=train_ds, 
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # 添加早停
    )
    
    trainer.train()
    trainer.save_model(f"{out_dir}/best")
    
    # 在测试集上评估
    results = trainer.evaluate(test_ds)
    print(f"{parent} 测试结果: {results}")
    
    # 保存评估结果
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/test_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

def main():
    df = load_level2_df()
    
    # 使用一致的数据分割
    full_train_df, full_val_df, full_test_df = get_consistent_split(df)
    
    os.makedirs(L2_OUT, exist_ok=True)
    
    # 按样本数量排序父类别，优先训练数据较多的类别
    parent_counts = full_train_df['level1'].value_counts()
    sorted_parents = parent_counts.index.tolist()
    
    for parent in sorted_parents:
        # 筛选出该父类别的数据
        train_df = full_train_df[full_train_df['level1'] == parent]
        val_df = full_val_df[full_val_df['level1'] == parent]
        test_df = full_test_df[full_test_df['level1'] == parent]
        
        # 合并训练集、验证集和测试集
        combined_df = pd.concat([train_df, val_df, test_df])
        train_one_parent(parent, combined_df)

if __name__ == "__main__":
    main()