import pandas as pd
import numpy as np
import torch
import time
import random
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from evaluate import load as load_metric
import warnings
#warnings.filterwarnings('ignore')

# 設定隨機種子確保結果可重現
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # 清理 GPU 快取
    torch.cuda.empty_cache()

# 分析數據分佈函數
def analyze_data_distribution(labels, title="數據分佈"):
    print(f"\n{title}:")
    label_names = ['relevance', 'concreteness', 'constructive']
    total_samples = len(labels)
    print(f"總樣本數: {total_samples}")
    
    for i, label_name in enumerate(label_names):
        positive_count = np.sum(labels[:, i])
        negative_count = total_samples - positive_count
        positive_ratio = positive_count / total_samples * 100
        print(f"{label_name}: 正樣本 {positive_count} ({positive_ratio:.1f}%), 負樣本 {negative_count} ({100-positive_ratio:.1f}%)")
    
    # 計算多標籤組合分佈
    unique_combinations, counts = np.unique(labels, axis=0, return_counts=True)
    print(f"\n多標籤組合數: {len(unique_combinations)}")
    print("標籤組合分佈 (relevance, concreteness, constructive):")
    for combo, count in zip(unique_combinations, counts):
        ratio = count / total_samples * 100
        print(f"  {tuple(combo.astype(int))}: {count} 樣本 ({ratio:.1f}%)")

# SMOTE 重新採樣函數
def apply_smote_to_sequences(input_ids, attention_masks, labels, random_state=42):
    try:
        from imblearn.over_sampling import SMOTE
        print("開始 SMOTE 重新採樣...")
        
        # 將多標籤轉換為組合標籤進行 SMOTE
        label_combinations = []
        for label_row in labels:
            combo_str = ''.join(map(str, label_row))
            label_combinations.append(int(combo_str, 2))  # 二進制轉十進制
        
        # 準備 SMOTE 輸入：將 input_ids 和 attention_masks 合併
        features = np.concatenate([input_ids, attention_masks], axis=1)
        
        smote = SMOTE(random_state=random_state)
        resampled_features, resampled_combo_labels = smote.fit_resample(features, label_combinations)
        
        # 分離 input_ids 和 attention_masks
        seq_len = input_ids.shape[1]
        resampled_input_ids = resampled_features[:, :seq_len]
        resampled_attention_masks = resampled_features[:, seq_len:]
        
        # 將組合標籤轉回多標籤格式
        resampled_labels = np.zeros((len(resampled_combo_labels), 3))
        for idx, combo_label in enumerate(resampled_combo_labels):
            binary_str = format(combo_label, '03b')  # 轉換為3位二進制
            for j, bit in enumerate(binary_str):
                resampled_labels[idx, j] = int(bit)
        
        print(f"原始樣本數: {len(labels)} -> 重新採樣後樣本數: {len(resampled_labels)}")
        print("SMOTE 重新採樣完成")
        
        return resampled_input_ids.astype(int), resampled_attention_masks.astype(int), resampled_labels.astype(int)
        
    except (ImportError, ValueError) as e:
        print(f"警告: SMOTE 重新採樣失敗 ({e})，使用原始數據")
        return input_ids, attention_masks, labels

# 1. 資料清理，已手動檢查過原始資料
def load_and_clean_data(path):
    data = pd.read_csv(path, encoding='utf-8')

    nan_rows = data[data.isna().any(axis=1)]
    if not nan_rows.empty:  
        print("下列資料含有 NaN，請檢查：")
        print(nan_rows.to_string(index=True))
    else:
        print("資料中沒有含 NaN 的列。")
    # 強化清理流程
    data['text'] = data['text'].apply(
        lambda x: str(x).replace('\n', ' ').replace('\r', '').strip() 
        if pd.notna(x) else ""
    )
    data = data[data['text'] != '']  # 移除空文字
    return data

data = load_and_clean_data('../data/cleaned_3label_data.csv') 
assert data.isna().sum().sum() == 0, "data存在未處理的 NaN！"
for col in ['relevance', 'concreteness', 'constructive']:
    data[col] = data[col].astype(int)

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=3,
    problem_type="multi_label_classification"
)

# 2. 定義分詞函數
def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=500)

labels = data[['relevance', 'concreteness', 'constructive']].values

# 分析原始數據分佈
analyze_data_distribution(labels, "原始數據分佈")

# 對數據集進行分詞
print("開始分詞處理...")
tokenized_inputs = tokenize_texts(data['text'].tolist())
all_input_ids = tokenized_inputs['input_ids'].numpy()
all_attention_mask = tokenized_inputs['attention_mask'].numpy()
print("分詞處理完成")

# 應用 SMOTE 重新採樣
all_input_ids, all_attention_mask, labels = apply_smote_to_sequences(
    all_input_ids, all_attention_mask, labels, random_state=42
)

# 分析重新採樣後的數據分佈
analyze_data_distribution(labels, "SMOTE 重新採樣後數據分佈")

# 3. 加載多標籤專用指標
precision_metric = load_metric("precision", config_name="multilabel")
recall_metric = load_metric("recall", config_name="multilabel")
f1_metric = load_metric("f1", config_name="multilabel")

# 4. 自訂加權損失
class WeightedBCELoss(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights)
        
    def forward(self, inputs, targets):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets.float(), 
            pos_weight=self.weights.to(inputs.device),
            reduction='none'
        )
        return loss.mean()

# 5. 多標籤資料集類別
class MultiLabelDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
        
    def __len__(self):
        return len(self.input_ids)

# 6. Hamming Score 函數
def hamming_score(y_true, y_pred):
    return np.mean([
        np.logical_and(t, p).sum() / np.logical_or(t, p).sum() if np.logical_or(t, p).sum() != 0 else 1.0
        for t, p in zip(y_true, y_pred)
    ])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = (predictions > 0.5).astype(int)
    
    precision_micro = precision_metric.compute(predictions=preds, references=labels, average='micro')['precision']
    recall_micro = recall_metric.compute(predictions=preds, references=labels, average='micro')['recall']
    f1_micro = f1_metric.compute(predictions=preds, references=labels, average='micro')['f1']
    
    precision_macro = precision_metric.compute(predictions=preds, references=labels, average='macro')['precision']
    recall_macro = recall_metric.compute(predictions=preds, references=labels, average='macro')['recall']
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']
    
    subset_acc = accuracy_score(labels, preds)
    hamming = hamming_score(labels, preds)
    
    return {
        "eval_subset_accuracy": subset_acc,
        "eval_hamming_score": hamming,
        "eval_precision_micro": precision_micro,
        "eval_recall_micro": recall_micro,
        "eval_f1_micro": f1_micro,
        "eval_precision_macro": precision_macro,
        "eval_recall_macro": recall_macro,
        "eval_f1_macro": f1_macro,
    }

class CustomTrainer(Trainer):
    def __init__(self, pos_weights, **kwargs):
        super().__init__(**kwargs)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pos_weights = torch.tensor(pos_weights).to(device)
        self.inference_times = []

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 重置 GPU 記憶體統計
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.perf_counter()
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        total_time = time.perf_counter() - start_time
        
        # 計算推理時間統計
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        num_samples = len(eval_dataloader.dataset)
        
        # 添加 GPU 記憶體使用量計算
        gpu_mem_usage = 0
        if torch.cuda.is_available():
            gpu_mem_usage = torch.cuda.max_memory_allocated() / 1e9  # 轉換為GB
        
        metrics.update({
            'avg_inference_time': total_time,
            'inference_time_per_sample': total_time / num_samples,
            'samples_per_second': num_samples / total_time,
            'gpu_mem_usage': gpu_mem_usage
        })
        
        return metrics
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 確保 pos_weights 與 logits 在同一設備
        pos_weights = self.pos_weights.to(logits.device)
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, 
            labels.float(),
            pos_weight=pos_weights
        )
        return (loss, outputs) if return_outputs else loss

# 8. 交叉驗證主流程
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []
best_trainer = None
best_f1 = 0.0

for fold, (train_idx, val_idx) in enumerate(kf.split(all_input_ids)):
    print(f"\n{'='*40}")
    print(f"Fold {fold+1}/10")
    print(f"{'='*40}")
    
    # 清理之前的 GPU 記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 分割資料
    train_inputs = all_input_ids[train_idx]
    train_masks = all_attention_mask[train_idx]
    train_labels = labels[train_idx]
    
    val_inputs = all_input_ids[val_idx]
    val_masks = all_attention_mask[val_idx]
    val_labels = labels[val_idx]

    # 動態類別權重計算
    pos_weights = []
    for i in range(3):
        n_pos = train_labels[:, i].sum()
        n_neg = len(train_labels) - n_pos
        pos_weights.append(n_neg / (n_pos + 1e-6))  # 避免除零

    # 模型初始化（每折獨立）
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=3,
        problem_type="multi_label_classification"
    )
    
    # 確保模型在正確的設備上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_dataset = MultiLabelDataset(train_inputs, train_masks, train_labels)
    val_dataset = MultiLabelDataset(val_inputs, val_masks, val_labels)

    # 動態閾值計算函數
    def find_optimal_threshold(predictions, labels):
        thresholds = []
        for i in range(3):
            pred_scores = predictions[:, i]
            best_thresh = 0.0
            best_f1 = 0.0
            for thresh in np.arange(0.1, 0.9, 0.05):
                f1 = f1_metric.compute(
                    predictions=(pred_scores > thresh).astype(int),
                    references=labels[:, i],
                    average='binary'
                )['f1']
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            thresholds.append(best_thresh)
        return np.array(thresholds)

    # 10. 訓練參數設定
    training_args = TrainingArguments(
        output_dir=f'./results_fold{fold}',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=6,
        weight_decay=0.001,
        logging_steps=50,
        save_strategy='no',
        report_to='none',
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # 11. 訓練器設定
    trainer = CustomTrainer(
        pos_weights=pos_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 12. 執行訓練
    trainer.train()
    
    # 13. 最終評估
    eval_results = trainer.evaluate()
    
    # 添加可解釋性分析
    sample_input = val_dataset[0]
    model.eval()
    
    # 確保輸入張量在正確的設備上
    input_ids = sample_input['input_ids'].unsqueeze(0).to(device)
    attention_mask = sample_input['attention_mask'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        # 計算平均注意力權重作為可解釋性指標
        attention_weights = outputs.attentions[-1].mean().item()
        eval_results['avg_attention_weight'] = attention_weights
    
    fold_results.append(eval_results)
    
    print(f"\nFold {fold+1} 評估結果:")
    print(f"Micro F1: {eval_results['eval_f1_micro']:.3f}")
    print(f"Macro F1: {eval_results['eval_f1_macro']:.3f}")
    print(f"推理時間: {eval_results['avg_inference_time']:.3f}s")
    print(f"每樣本推理時間: {eval_results['inference_time_per_sample']:.6f}s")
    print(f"GPU記憶體使用: {eval_results['gpu_mem_usage']:.3f}GB")
    print(f"注意力權重: {eval_results['avg_attention_weight']:.3f}")

    # 保存最佳模型
    if eval_results['eval_f1_macro'] > best_f1:
        best_f1 = eval_results['eval_f1_macro']
        if best_trainer is not None:
            del best_trainer
        best_trainer = trainer
        best_model = model
        print(f"新的最佳模型! Macro F1: {best_f1:.3f}")
    else:
        del model, trainer
    
    # 清理數據集
    del train_dataset, val_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 14. 交叉驗證結果匯總
print("\n\n交叉驗證最終結果:")
avg_metrics = {
    'micro_f1': np.mean([r['eval_f1_micro'] for r in fold_results]),
    'macro_f1': np.mean([r['eval_f1_macro'] for r in fold_results]),
    'micro_precision': np.mean([r['eval_precision_micro'] for r in fold_results]),
    'macro_precision': np.mean([r['eval_precision_macro'] for r in fold_results]),
    'micro_recall': np.mean([r['eval_recall_micro'] for r in fold_results]),
    'macro_recall': np.mean([r['eval_recall_macro'] for r in fold_results]),
    'avg_inference_time': np.mean([r['avg_inference_time'] for r in fold_results]),
    'avg_gpu_memory': np.mean([r['gpu_mem_usage'] for r in fold_results]),
    'avg_attention_weight': np.mean([r.get('avg_attention_weight', 0) for r in fold_results])
}

print(f"Average Micro F1: {avg_metrics['micro_f1']:.3f}")
print(f"Average Macro F1: {avg_metrics['macro_f1']:.3f}")
print(f"Average Micro Precision: {avg_metrics['micro_precision']:.3f}")
print(f"Average Macro Precision: {avg_metrics['macro_precision']:.3f}")
print(f"Average Micro Recall: {avg_metrics['micro_recall']:.3f}")
print(f"Average Macro Recall: {avg_metrics['macro_recall']:.3f}")
print(f"Average Inference Time: {avg_metrics['avg_inference_time']:.3f}s")
print(f"Average GPU Memory Usage: {avg_metrics['avg_gpu_memory']:.3f}GB")
print(f"Average Attention Weight: {avg_metrics['avg_attention_weight']:.6f}")

# 保存最佳模型
if best_trainer is not None:
    best_trainer.save_model("./bert_3label_finetuned_model")
    tokenizer.save_pretrained("./bert_3label_finetuned_model")
    print(f"\n最佳模型已保存! (Macro F1: {best_f1:.3f})")
else:
    print("\n警告: 沒有找到可保存的模型")

# 記憶體-效能關係圖
plt.figure(figsize=(15, 5))

# 子圖1: 記憶體vs效能
plt.subplot(1, 3, 1)
plt.scatter(
    [r['gpu_mem_usage'] for r in fold_results],
    [r['eval_f1_macro'] for r in fold_results],
    c=[r['avg_inference_time'] for r in fold_results],
    cmap='viridis',
    s=100
)
plt.colorbar(label='Inference Time (s)')
plt.xlabel('GPU Memory Usage (GB)')
plt.ylabel('Macro F1 Score')
plt.title('Memory vs Performance (BERT)')

# 子圖2: 推理時間vs效能
plt.subplot(1, 3, 2)
plt.scatter(
    [r['avg_inference_time'] for r in fold_results],
    [r['eval_f1_macro'] for r in fold_results],
    c=range(len(fold_results)),
    cmap='tab10'
)
plt.xlabel('Inference Time (s)')
plt.ylabel('Macro F1 Score')
plt.title('Inference Time vs Performance (BERT)')

# 子圖3: 各fold效能比較
plt.subplot(1, 3, 3)
folds = range(1, len(fold_results) + 1)
plt.plot(folds, [r['eval_f1_macro'] for r in fold_results], 'bo-', label='Macro F1')
plt.plot(folds, [r['eval_f1_micro'] for r in fold_results], 'ro-', label='Micro F1')
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('Performance Across Folds (BERT)')
plt.legend()

plt.tight_layout()
plt.savefig('./bert_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 可解釋性報告
print("\n可解釋性分析:")
avg_attention = avg_metrics['avg_attention_weight']
print(f"平均注意力權重: {avg_attention:.6f}")
print("注意力權重較高表示模型對輸入文本的關注度較高")

# 添加詳細的注意力權重分析
attention_weights = [r.get('avg_attention_weight', 0) for r in fold_results]
print(f"注意力權重標準差: {np.std(attention_weights):.6f}")
print(f"注意力權重範圍: {np.min(attention_weights):.6f} - {np.max(attention_weights):.6f}")
