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
#warnings.filterwarnings('ignore')  # 可以刪掉

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

    nan_rows = data[data.isna().any(axis=1)]  # 把 data 裡有 NaN 或 None 的所有資料列 找出來並存成 nan_rows
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
assert data.isna().sum().sum() == 0, "data存在未處理的 NaN！"  # 檢查是否有缺失值 data.isna()：找出所有 NaN 的位置。總數不是 0，程式就會報
for col in ['relevance', 'concreteness', 'constructive']:
    data[col] = data[col].astype(int)

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')  # 自動下載並建立對應的分詞器
model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=3,
        problem_type="multi_label_classification"
    )

# 2. 定義分詞函數
def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=500)  # 輸出 PyTorch tensor 格式

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
f1_metric = load_metric("f1", config_name="multilabel")  # 新增這行

# 4. 自訂加權損失
class WeightedBCELoss(torch.nn.Module):  # 透過 pos_weight 參數給予少數類別更高的權重，強制模型更關注少數類別的學習
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


# 7. 資料集轉換
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

def hamming_score(y_true, y_pred):
    return np.mean([
        np.logical_and(t, p).sum() / np.logical_or(t, p).sum() if np.logical_or(t, p).sum() != 0 else 1.0
        for t, p in zip(y_true, y_pred)
    ])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = (predictions > 0.5).astype(int)  # 暫時使用固定閾值
    
    precision_micro = precision_metric.compute(predictions=preds, references=labels, average='micro')['precision']
    recall_micro = recall_metric.compute(predictions=preds, references=labels, average='micro')['recall']
    f1_micro = f1_metric.compute(predictions=preds, references=labels, average='micro')['f1']
    
    precision_macro = precision_metric.compute(predictions=preds, references=labels, average='macro')['precision']
    recall_macro = recall_metric.compute(predictions=preds, references=labels, average='macro')['recall']
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']
    
    subset_acc = accuracy_score(labels, preds)
    hamming = hamming_score(labels, preds)
    
    # 修正 GPU 記憶體使用量計算
    gpu_mem_usage = 0
    if torch.cuda.is_available():
        gpu_mem_usage = torch.cuda.max_memory_allocated() / 1e9  # 轉換為GB
    
    return {
        "eval_subset_accuracy": subset_acc,
        "eval_hamming_score": hamming,
        "eval_precision_micro": precision_micro,
        "eval_recall_micro": recall_micro,
        "eval_f1_micro": f1_micro,
        "eval_precision_macro": precision_macro,
        "eval_recall_macro": recall_macro,
        "eval_f1_macro": f1_macro,
        "gpu_mem_usage": gpu_mem_usage
    }

class CustomTrainer(Trainer):
    def __init__(self, pos_weights, **kwargs):
        super().__init__(**kwargs)
        # 確保 pos_weights 在正確的設備上
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
            'gpu_mem_usage': gpu_mem_usage  # 添加這行
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

# 2. 分析數據分佈函數
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

# 3. SMOTE 重新採樣函數
def apply_smote_to_sequences(input_ids, attention_masks, labels, random_state=42):
    """
    使用 SMOTE 對已分詞的序列數據進行重新採樣
    """
    print("開始 SMOTE 重新採樣...")
    
    # 將多標籤轉換為組合標籤進行 SMOTE
    label_combinations = []
    for label_row in labels:
        combo_str = ''.join(map(str, label_row))
        label_combinations.append(int(combo_str, 2))  # 二進制轉十進制
    
    # 統計原始組合分佈
    unique_combos, combo_counts = np.unique(label_combinations, return_counts=True)
    print(f"原始標籤組合分佈:")
    for combo, count in zip(unique_combos, combo_counts):
        binary_str = format(combo, '03b')
        print(f"  {binary_str}: {count} 樣本")
    
    # 準備 SMOTE 輸入：將 input_ids 和 attention_masks 合併
    features = np.concatenate([input_ids, attention_masks], axis=1)
    
    try:
        smote = SMOTE(random_state=random_state, k_neighbors=min(5, min(combo_counts)-1))
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
        
        # 統計重新採樣後的組合分佈
        unique_combos_new, combo_counts_new = np.unique(resampled_combo_labels, return_counts=True)
        print(f"\nSMOTE 重新採樣後的標籤組合分佈:")
        for combo, count in zip(unique_combos_new, combo_counts_new):
            binary_str = format(combo, '03b')
            print(f"  {binary_str}: {count} 樣本")
        
        print(f"原始樣本數: {len(labels)} -> 重新採樣後樣本數: {len(resampled_labels)}")
        print("SMOTE 重新採樣完成")
        
        return resampled_input_ids.astype(int), resampled_attention_masks.astype(int), resampled_labels.astype(int)
        
    except ValueError as e:
        print(f"警告: SMOTE 重新採樣失敗 ({e})，使用原始數據")
        return input_ids, attention_masks, labels

# 4. 原有的資料清理函數
def load_and_clean_data(path):
    data = pd.read_csv(path, encoding='utf-8')

    nan_rows = data[data.isna().any(axis=1)]  # 把 data 裡有 NaN 或 None 的所有資料列 找出來並存成 nan_rows
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
assert data.isna().sum().sum() == 0, "data存在未處理的 NaN！"  # 檢查是否有缺失值 data.isna()：找出所有 NaN 的位置。總數不是 0，程式就會報
for col in ['relevance', 'concreteness', 'constructive']:
    data[col] = data[col].astype(int)

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')  # 自動下載並建立對應的分詞器
model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=3,
        problem_type="multi_label_classification"
    )

# 2. 定義分詞函數
def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=500)  # 輸出 PyTorch tensor 格式

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
f1_metric = load_metric("f1", config_name="multilabel")  # 新增這行

# 4. 自訂加權損失
class WeightedBCELoss(torch.nn.Module):  # 透過 pos_weight 參數給予少數類別更高的權重，強制模型更關注少數類別的學習
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


# 7. 資料集轉換
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

def hamming_score(y_true, y_pred):
    return np.mean([
        np.logical_and(t, p).sum() / np.logical_or(t, p).sum() if np.logical_or(t, p).sum() != 0 else 1.0
        for t, p in zip(y_true, y_pred)
    ])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = (predictions > 0.5).astype(int)  # 暫時使用固定閾值
    
    precision_micro = precision_metric.compute(predictions=preds, references=labels, average='micro')['precision']
    recall_micro = recall_metric.compute(predictions=preds, references=labels, average='micro')['recall']
    f1_micro = f1_metric.compute(predictions=preds, references=labels, average='micro')['f1']
    
    precision_macro = precision_metric.compute(predictions=preds, references=labels, average='macro')['precision']
    recall_macro = recall_metric.compute(predictions=preds, references=labels, average='macro')['recall']
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']
    
    subset_acc = accuracy_score(labels, preds)
    hamming = hamming_score(labels, preds)
    
    # 修正 GPU 記憶體使用量計算
    gpu_mem_usage = 0
    if torch.cuda.is_available():
        gpu_mem_usage = torch.cuda.max_memory_allocated() / 1e9  # 轉換為GB
    
    return {
        "eval_subset_accuracy": subset_acc,
        "eval_hamming_score": hamming,
        "eval_precision_micro": precision_micro,
        "eval_recall_micro": recall_micro,
        "eval_f1_micro": f1_micro,
        "eval_precision_macro": precision_macro,
        "eval_recall_macro": recall_macro,
        "eval_f1_macro": f1_macro,
        "gpu_mem_usage": gpu_mem_usage
    }

class CustomTrainer(Trainer):
    def __init__(self, pos_weights, **kwargs):
        super().__init__(**kwargs)
        # 確保 pos_weights 在正確的設備上
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
            'gpu_mem_usage': gpu_mem_usage  # 添加這行
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

# 2. 分析數據分佈函數
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

# 3. SMOTE 重新採樣函數
def apply_smote_to_sequences(input_ids, attention_masks, labels, random_state=42):
    """
    使用 SMOTE 對已分詞的序列數據進行重新採樣
    """
    print("開始 SMOTE 重新採樣...")
    
    # 將多標籤轉換為組合標籤進行 SMOTE
    label_combinations = []
    for label_row in labels:
        combo_str = ''.join(map(str, label_row))
        label_combinations.append(int(combo_str, 2))  # 二進制轉十進制
    
    # 統計原始組合分佈
    unique_combos, combo_counts = np.unique(label_combinations, return_counts=True)
    print(f"原始標籤組合分佈:")
    for combo, count in zip(unique_combos, combo_counts):
        binary_str = format(combo, '03b')
        print(f"  {binary_str}: {count} 樣本")
    
    # 準備 SMOTE 輸入：將 input_ids 和 attention_masks 合併
    features = np.concatenate([input_ids, attention_masks], axis=1)
    
    try:
        smote = SMOTE(random_state=random_state, k_neighbors=min(5, min(combo_counts)-1))
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
        
        # 統計重新採樣後的組合分佈
        unique_combos_new, combo_counts_new = np.unique(resampled_combo_labels, return_counts=True)
        print(f"\nSMOTE 重新採樣後的標籤組合分佈:")
        for combo, count in zip(unique_combos_new, combo_counts_new):
            binary_str = format(combo, '03b')
            print(f"  {binary_str}: {count} 樣本")
        
        print(f"原始樣本數: {len(labels)} -> 重新採樣後樣本數: {len(resampled_labels)}")
        print("SMOTE 重新採樣完成")
        
        return resampled_input_ids.astype(int), resampled_attention_masks.astype(int), resampled_labels.astype(int)
        
    except ValueError as e:
        print(f"警告: SMOTE 重新採樣失敗 ({e})，使用原始數據")
        return input_ids, attention_masks, labels

# 4. 原有的資料清理函數
def load_and_clean_data(path):
    data = pd.read_csv(path, encoding='utf-8')

    nan_rows = data[data.isna().any(axis=1)]  # 把 data 裡有 NaN 或 None 的所有資料列 找出來並存成 nan_rows
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
assert data.isna().sum().sum() == 0, "data存在未處理的 NaN！"  # 檢查是否有缺失值 data.isna()：找出所有 NaN 的位置。總數不是 0，程式就會報
for col in ['relevance', 'concreteness', 'constructive']:
    data[col] = data[col].astype(int)

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')  # 自動下載並建立對應的分詞器
model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=3,
        problem_type="multi_label_classification"
    )

# 2. 定義分詞函數
def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=500)  # 輸出 PyTorch tensor 格式

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
f1_metric = load_metric("f1", config_name="multilabel")  # 新增這行

# 4. 自訂加權損失
class WeightedBCELoss(torch.nn.Module):  # 透過 pos_weight 參數給予少數類別更高的權重，強制模型更關注少數類別的學習
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


# 7. 資料集轉換
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

def hamming_score(y_true, y_pred):
    return np.mean([
        np.logical_and(t, p).sum() / np.logical_or(t, p).sum() if np.logical_or(t, p).sum() != 0 else 1.0
        for t, p in zip(y_true, y_pred)
    ])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = (predictions > 0.5).astype(int)  # 暫時使用固定閾值
    
    precision_micro = precision_metric.compute(predictions=preds, references=labels, average='micro')['precision']
    recall_micro = recall_metric.compute(predictions=preds, references=labels, average='micro')['recall']
    f1_micro = f1_metric.compute(predictions=preds, references=labels, average='micro')['f1']
    
    precision_macro = precision_metric.compute(predictions=preds, references=labels, average='macro')['precision']
    recall_macro = recall_metric.compute(predictions=preds, references=labels, average='macro')['recall']
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']
    
    subset_acc = accuracy_score(labels, preds)
    hamming = hamming_score(labels, preds)
    
    # 修正 GPU 記憶體使用量計算
    gpu_mem_usage = 0
    if torch.cuda.is_available():
        gpu_mem_usage = torch.cuda.max_memory_allocated() / 1e9  # 轉換為GB
    
    return {
        "eval_subset_accuracy": subset_acc,
        "eval_hamming_score": hamming,
        "eval_precision_micro": precision_micro,
        "eval_recall_micro": recall_micro,
        "eval_f1_micro": f1_micro,
        "eval_precision_macro": precision_macro,
        "eval_recall_macro": recall_macro,
        "eval_f1_macro": f1_macro,
        "gpu_mem_usage": gpu_mem_usage
    }

class CustomTrainer(Trainer):
    def __init__(self, pos_weights, **kwargs):
        super().__init__(**kwargs)
        # 確保 pos_weights 在正確的設備上
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
            'gpu_mem_usage': gpu_mem_usage  # 添加這行
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

# 2. 分析數據分佈函數
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

# 3. SMOTE 重新採樣函數
def apply_smote_to_sequences(input_ids, attention_masks, labels, random_state=42):
    """
    使用 SMOTE 對已分詞的序列數據進行重新採樣
    """
    print("開始 SMOTE 重新採樣...")
    
    # 將多標籤轉換為組合標籤進行 SMOTE
    label_combinations = []
    for label_row in labels:
        combo_str = ''.join(map(str, label_row))
        label_combinations.append(int(combo_str, 2))  # 二進制轉十進制
    
    # 統計原始組合分佈
    unique_combos, combo_counts = np.unique(label_combinations, return_counts=True)
    print(f"原始標籤組合分佈:")
    for combo, count in zip(unique_combos, combo_counts):
        binary_str = format(combo, '03b')
        print(f"  {binary_str}: {count} 樣本")
    
    # 準備 SMOTE 輸入：將 input_ids 和 attention_masks 合併
    features = np.concatenate([input_ids, attention_masks], axis=1)
    
    try:
        smote = SMOTE(random_state=random_state, k_neighbors=min(5, min(combo_counts)-1))
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
        
        # 統計重新採樣後的組合分佈
        unique_combos_new, combo_counts_new = np.unique(resampled_combo_labels, return_counts=True)
        print(f"\nSMOTE 重新採樣後的標籤組合分佈:")
        for combo, count in zip(unique_combos_new, combo_counts_new):
            binary_str = format(combo, '03b')
            print(f"  {binary_str}: {count} 樣本")
        
        print(f"原始樣本數: {len(labels)} -> 重新採樣後樣本數: {len(resampled_labels)}")
        print("SMOTE 重新採樣完成")
        
        return resampled_input_ids.astype(int), resampled_attention_masks.astype(int), resampled_labels.astype(int)
        
    except ValueError as e:
        print(f"警告: SMOTE 重新採樣失敗 ({e})，使用原始數據")
        return input_ids, attention_masks, labels

# 4. 原有的資料清理函數
def load_and_clean_data(path):
    data = pd.read_csv(path, encoding='utf-8')

    nan_rows = data[data.isna().any(axis=1)]  # 把 data 裡有 NaN 或 None 的所有資料列 找出來並存成 nan_rows
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
assert data.isna().sum().sum() == 0, "data存在未處理的 NaN！"  # 檢查是否有缺失值 data.isna()：找出所有 NaN 的位置。總數不是 0，程式就會報
for col in ['relevance', 'concreteness', 'constructive']:
    data[col] = data[col].astype(int)

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')  # 自動下載並建立對應的分詞器
model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=3,
        problem_type="multi_label_classification"
    )

# 2. 定義分詞函數
def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=500)  # 輸出 PyTorch tensor 格式

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
f1_metric = load_metric("f1", config_name="multilabel")  # 新增這行

# 4. 自訂加權損失
class WeightedBCELoss(torch.nn.Module):  # 透過 pos_weight 參數給予少數類別更高的權重，強制模型更關注少數類別的學習
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


# 7. 資料集轉換
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

def hamming_score(y_true, y_pred):
    return np.mean([
        np.logical_and(t, p).sum() / np.logical_or(t, p).sum() if np.logical_or(t, p).sum() != 0 else 1.0
        for t, p in zip(y_true, y_pred)
    ])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = (predictions > 0.5).astype(int)  # 暫時使用固定閾值
    
    precision_micro = precision_metric.compute(predictions=preds, references=labels, average='micro')['precision']
    recall_micro = recall_metric.compute(predictions=preds, references=labels, average='micro')['recall']
    f1_micro = f1_metric.compute(predictions=preds, references=labels, average='micro')['f1']
    
    precision_macro = precision_metric.compute(predictions=preds, references=labels, average='macro')['precision']
    recall_macro = recall_metric.compute(predictions=preds, references=labels, average='macro')['recall']
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']
    
    subset_acc = accuracy_score(labels, preds)
    hamming = hamming_score(labels, preds)
    
    # 修正 GPU 記憶體使用量計算
    gpu_mem_usage = 0
    if torch.cuda.is_available():
        gpu_mem_usage = torch.cuda.max_memory_allocated() / 1e9  # 轉換為GB
    
    return {
        "eval_subset_accuracy": subset_acc,
        "eval_hamming_score": hamming,
        "eval_precision_micro": precision_micro,
        "eval_recall_micro": recall_micro,
        "eval_f1_micro": f1_micro,
        "eval_precision_macro": precision_macro,
        "eval_recall_macro": recall_macro,
        "eval_f1_macro": f1_macro,
        "gpu_mem_usage": gpu_mem_usage
    }

class CustomTrainer(Trainer):
    def __init__(self, pos_weights, **kwargs):
        super().__init__(**kwargs)
        # 確保 pos_weights 在正確的設備上
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
            'gpu_mem_usage': gpu_mem_usage  # 添加這行
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

# 2. 分析數據分佈函數
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

# 3. SMOTE 重新採樣函數
def apply_smote_to_sequences(input_ids, attention_masks, labels, random_state=42):
    """
    使用 SMOTE 對已分詞的序列數據進行重新採樣
    """
    print("開始 SMOTE 重新採樣...")
    
    # 將多標籤轉換為組合標籤進行 SMOTE
    label_combinations = []
    for label_row in labels:
        combo_str = ''.join(map(str, label_row))
        label_combinations.append(int(combo_str, 2))  # 二進制轉十進制
    
    # 統計原始組合分佈
    unique_combos, combo_counts = np.unique(label_combinations, return_counts=True)
    print(f"原始標籤組合分佈:")
    for combo, count in zip(unique_combos, combo_counts):
        binary_str = format(combo, '03b')
        print(f"  {binary_str}: {count} 樣本")
    
    # 準備 SMOTE 輸入：將 input_ids 和 attention_masks 合併
    features = np.concatenate([input_ids, attention_masks], axis=1)
    
    try:
        smote = SMOTE(random_state=random_state, k_neighbors=min(5, min(combo_counts)-1))
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
        
        # 統計重新採樣後的組合分佈
        unique_combos_new, combo_counts_new = np.unique(resampled_combo_labels, return_counts=True)
        print(f"\nSMOTE 重新採樣後的標籤組合分佈:")
        for combo, count in zip(unique_combos_new, combo_counts_new):
            binary_str = format(combo, '03b')
            print(f"  {binary_str}: {count} 樣本")
        
        print(f"原始樣本數: {len(labels)} -> 重新採樣後樣本數: {len(resampled_labels)}")
        print("SMOTE 重新採樣完成")
        
        return resampled_input_ids.astype(int), resampled_attention_masks.astype(int), resampled_labels.astype(int)
        
    except ValueError as e:
        print(f"警告: SMOTE 重新採樣失敗 ({e})，使用原始數據")
        return input_ids, attention_masks, labels

# 4. 原有的資料清理函數
def load_and_clean_data(path):
    data = pd.read_csv(path, encoding='utf-8')

    nan_rows = data[data.isna().any(axis=1)]  # 把 data 裡有 NaN 或 None 的所有資料列 找出來並存成 nan_rows
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
assert data.isna().sum().sum() == 0, "data存在未處理的 NaN！"  # 檢查是否有缺失值 data.isna()：找出所有 NaN 的位置。總數不是 0，程式就會報
for col in ['relevance', 'concreteness', 'constructive']:
    data[col] = data[col].astype(int)

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')  # 自動下載並建立對應的分詞器
model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=3,
        problem_type="multi_label_classification"
    )

# 2. 定義分詞函數
def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=500)  # 輸出 PyTorch tensor 格式

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
f1_metric = load_metric("f1", config_name="multilabel")  # 新增這行

# 4. 自訂加權損失
class WeightedBCELoss(torch.nn.Module):  # 透過 pos_weight 參數給予少數類別更高的權重，強制模型更關注少數類別的學習
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


# 7. 資料集轉換
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

def hamming_score(y_true, y_pred):
    return np.mean([
        np.logical_and(t, p).sum() / np.logical_or(t, p).sum() if np.logical_or(t, p).sum() != 0 else 1.0
        for t, p in zip(y_true, y_pred)
    ])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = (predictions > 0.5).astype(int)  # 暫時使用固定閾值
    
    precision_micro = precision_metric.compute(predictions=preds, references=labels, average='micro')['precision']
    recall_micro = recall_metric.compute(predictions=preds, references=labels, average='micro')['recall']
    f1_micro = f1_metric.compute(predictions=preds, references=labels, average='micro')['f1']
    
    precision_macro = precision_metric.compute(predictions=preds, references=labels, average='macro')['precision']
    recall_macro = recall_metric.compute(predictions=preds, references=labels, average='macro')['recall']
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']
    
    subset_acc = accuracy_score(labels, preds)
    hamming = hamming_score(labels, preds)
    
    # 修正 GPU 記憶體使用量計算
    gpu_mem_usage = 0
    if torch.cuda.is_available():
        gpu_mem_usage = torch.cuda.max_memory_allocated() / 1e9  # 轉換為GB
    
    return {
        "eval_subset_accuracy": subset_acc,
        "eval_hamming_score": hamming,
        "eval_precision_micro": precision_micro,
        "eval_recall_micro": recall_micro,
        "eval_f1_micro": f1_micro,
        "eval_precision_macro": precision_macro,
        "eval_recall_macro": recall_macro,
        "eval_f1_macro": f1_macro,
        "gpu_mem_usage": gpu_mem_usage
    }

class CustomTrainer(Trainer):
    def __init__(self, pos_weights, **kwargs):
        super().__init__(**kwargs)
        # 確保 pos_weights 在正確的設備上
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
            'gpu_mem_usage': gpu_mem_usage  # 添加這行
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

# 2. 分析數據分佈函數
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

# 3. SMOTE 重新採樣函數
def apply_smote_to_sequences(input_ids, attention_masks, labels, random_state=42):
    """
    使用 SMOTE 對已分詞的序列數據進行重新採樣
    """
    print("開始 SMOTE 重新採樣...")
    
    # 將多標籤轉換為組合標籤進行 SMOTE
    label_combinations = []
    for label_row in labels:
        combo_str = ''.join(map(str, label_row))
        label_combinations.append(int(combo_str, 2))  # 二進制轉十進制
    
    # 統計原始組合分佈
    unique_combos, combo_counts = np.unique(label_combinations, return_counts=True)
    print(f"原始標籤組合分佈:")
    for combo, count in zip(unique_combos, combo_counts):
        binary_str = format(combo, '03b')
        print(f"  {binary_str}: {count} 樣本")
    
    # 準備 SMOTE 輸入：將 input_ids 和 attention_masks 合併
    features = np.concatenate([input_ids, attention_masks], axis=1)
    
    try:
        smote = SMOTE(random_state=random_state, k_neighbors=min(5, min(combo_counts)-1))
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
        
        # 統計重新採樣後的組合分佈
        unique_combos_new, combo_counts_new = np.unique(resampled_combo_labels, return_counts=True)
        print(f"\nSMOTE 重新採樣後的標籤組合分佈:")
        for combo, count in zip(unique_combos_new, combo_counts_new):
            binary_str = format(combo, '03b')
            print(f"  {binary_str}: {count} 樣本")
        
        print(f"原始樣本數: {len(labels)} -> 重新採樣後樣本數: {len(resampled_labels)}")
        print("SMOTE 重新採樣完成")
        
        return resampled_input_ids.astype(int), resampled_attention_masks.astype(int), resampled_labels.astype(int)
        
    except ValueError as e:
        print(f"警告: SMOTE 重新採樣失敗 ({e})，使用原始數據")
        return input_ids, attention_masks, labels

# 4. 原有的資料清理函數
def load_and_clean_data(path):
    data = pd.read_csv(path, encoding='utf-8')

    nan_rows = data[data.isna().any(axis=1)]  # 把 data 裡有 NaN 或 None 的所有資料列 找出來並存成 nan_rows
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
assert data.isna().sum().sum() == 0, "data存在未處理的 NaN！"  # 檢查是否有缺失值 data.isna()：找出所有 NaN 的位置。總數不是 0，程式就會報
for col in ['relevance', 'concreteness', 'constructive']:
    data[col] = data[col].astype(int)

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')  # 自動下載並建立對應的分詞器
model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=3,
        problem_type="multi_label_classification"
    )

# 2. 定義分詞函數
def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=500)  # 輸出 PyTorch tensor 格式

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
f1_metric = load_metric("f1", config_name="multilabel")  # 新增這行

# 4. 自訂加權損失
class WeightedBCELoss(torch.nn.Module):  # 透過 pos_weight 參數給予少數類別更高的權重，強制模型更關注少數類別的學習
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


# 7. 資料集轉換
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

def hamming_score(y_true, y_pred):
    return np.mean([
        np.logical_and(t, p).sum() / np.logical_or(t, p).sum() if np.logical_or(t, p).sum() != 0 else 1.0
        for t, p in zip(y_true, y_pred)
    ])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = (predictions > 0.5).astype(int)  # 暫時使用固定閾值
    
    precision_micro = precision_metric.compute(predictions=preds, references=labels, average='micro')['precision']
    recall_micro = recall_metric.compute(predictions=preds, references=labels, average='micro')['recall']
    f1_micro = f1_metric.compute(predictions=preds, references=labels, average='micro')['f1']
    
    precision_macro = precision_metric.compute(predictions=preds, references=labels, average='macro')['precision']
    recall_macro = recall_metric.compute(predictions=preds, references=labels, average='macro')['recall']
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']
    
    subset_acc = accuracy_score(labels, preds)
    hamming = hamming_score(labels, preds)
    
    # 修正 GPU 記憶體使用量計算
    gpu_mem_usage = 0
    if torch.cuda.is_available():
        gpu_mem_usage = torch.cuda.max_memory_allocated() / 1e9  # 轉換為GB
    
    return {
        "eval_subset_accuracy": subset_acc,
        "eval_hamming_score": hamming,
        "eval_precision_micro": precision_micro,
        "eval_recall_micro": recall_micro,
        "eval_f1_micro": f1_micro,
        "eval_precision_macro": precision_macro,
        "eval_recall_macro": recall_macro,
        "eval_f1_macro": f1_macro,
        "gpu_mem_usage": gpu_mem_usage
    }

class CustomTrainer(Trainer):
    def __init__(self, pos_weights, **kwargs):
        super().__init__(**kwargs)
        # 確保 pos_weights 在正確的設備上
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
            'gpu_mem_usage': gpu_mem_usage  # 添加這行
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
