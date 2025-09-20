import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import warnings
import jieba
import pickle
from collections import Counter
import re
from imblearn.over_sampling import SMOTE
from collections import Counter
#warnings.filterwarnings('ignore')

# 設定隨機種子確保結果可重現
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()

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

# 2. 中文文本預處理
def preprocess_chinese_text(text):
    """中文文本預處理：分詞"""
    words = jieba.cut(text)
    return list(words)

print("開始文本預處理...")
data['processed_text'] = data['text'].apply(preprocess_chinese_text)
print("文本預處理完成")

labels = data[['relevance', 'concreteness', 'constructive']].values

# 2.5 數據分佈分析函數
def analyze_data_distribution(labels, title="數據分佈"):
    """分析並顯示多標籤數據分佈"""
    total_samples = len(labels)
    print(f"\n{title}:")
    print(f"總樣本數: {total_samples}")
    
    # 單標籤分佈
    label_names = ['relevance', 'concreteness', 'constructive']
    for i, label_name in enumerate(label_names):
        positive = labels[:, i].sum()
        negative = total_samples - positive
        pos_pct = positive / total_samples * 100
        neg_pct = negative / total_samples * 100
        print(f"{label_name}: 正樣本 {positive} ({pos_pct:.1f}%), 負樣本 {negative} ({neg_pct:.1f}%)")
    
    # 多標籤組合分佈
    label_combinations = {}
    for sample in labels:
        combo = tuple(sample)
        label_combinations[combo] = label_combinations.get(combo, 0) + 1
    
    print(f"\n多標籤組合數: {len(label_combinations)}")
    print("標籤組合分佈 (relevance, concreteness, constructive):")
    for combo, count in sorted(label_combinations.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_samples * 100
        print(f"  {combo}: {count} 樣本 ({percentage:.1f}%)")

def apply_smote_to_sequences(sequences, labels, random_state=42):
    """對序列數據應用 SMOTE 重新採樣"""
    # 將多標籤轉換為單一組合標籤用於 SMOTE
    label_combinations = {}
    combo_labels = []
    
    for i, sample in enumerate(labels):
        combo = tuple(sample)
        if combo not in label_combinations:
            label_combinations[combo] = len(label_combinations)
        combo_labels.append(label_combinations[combo])
    
    combo_labels = np.array(combo_labels)
    
    # 檢查最少樣本數的類別
    combo_counts = Counter(combo_labels)
    min_samples = min(combo_counts.values())
    
    print(f"最少樣本數的類別有 {min_samples} 個樣本")
    
    if min_samples < 2:
        print(f"警告: 最少類別樣本數 ({min_samples}) 小於2，無法進行SMOTE，使用原始數據")
        return sequences, labels, False
    
    try:
        # 調整 k_neighbors 參數
        k_neighbors = min(min_samples - 1, 5)
        print(f"調整 k_neighbors 為 {k_neighbors}")
        
        # 應用 SMOTE
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        sequences_resampled, combo_labels_resampled = smote.fit_resample(sequences, combo_labels)
        
        # 重新構建多標籤
        combo_to_multilabel = {v: k for k, v in label_combinations.items()}
        labels_resampled = np.array([combo_to_multilabel[combo] for combo in combo_labels_resampled])
        
        print(f"原始樣本數: {len(sequences)} -> 重新採樣後樣本數: {len(sequences_resampled)}")
        print("SMOTE 重新採樣完成")
        
        return sequences_resampled, labels_resampled, True
        
    except Exception as e:
        print(f"SMOTE 執行失敗: {e}")
        print("使用原始數據")
        return sequences, labels, False

# 原始數據分佈分析
analyze_data_distribution(labels, "原始數據分佈")

# 計算原始標籤分佈統計
total_samples = len(labels)
label_names = ['相關性標籤', '具體性標籤', '建設性標籤']
label_cols = ['relevance', 'concreteness', 'constructive']

print(f"\n{'='*60}")
print("原始資料標籤分佈統計")
print(f"{'='*60}")
print(f"總評論數: {total_samples}")
print(f"{'標籤類型':<12} {'出現次數':<8} {'比例':<8} {'分佈圖'}")
print("-" * 60)

for i, (label_name, col) in enumerate(zip(label_names, label_cols)):
    count = labels[:, i].sum()
    percentage = count / total_samples * 100
    bar_length = int(percentage / 2)  # 每個字符代表2%
    bar = '█' * bar_length + '░' * (50 - bar_length)
    print(f"{label_name:<12} {count:<8} {percentage:>6.1f}%   {bar}")

print("-" * 60)
print("圖例: █ = 2%, ░ = 未達到")

labels = data[['relevance', 'concreteness', 'constructive']].values

# 3. 構建詞彙表和詞向量
def build_vocab(texts, min_freq=2):
    """構建詞彙表"""
    word_counts = Counter()
    for text in texts:
        word_counts.update(text)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def load_word_embeddings(vocab, embedding_dim=100):
    """初始化詞向量"""
    embeddings = np.random.normal(0, 0.1, (len(vocab), embedding_dim))
    # 將 PAD 設為零向量
    embeddings[0] = np.zeros(embedding_dim)
    return embeddings

def texts_to_sequences(texts, vocab, max_len=200):
    """將文本轉換為數字序列"""
    sequences = []
    for text in texts:
        seq = [vocab.get(word, vocab['<UNK>']) for word in text]
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [0] * (max_len - len(seq))  # padding
        sequences.append(seq)
    return np.array(sequences)

# 4. CNN + Attention 模型定義
class CNNAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_labels, 
                 embeddings=None, dropout=0.3):
        super(CNNAttentionClassifier, self).__init__()
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        
        # 多尺度 CNN 層
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        
        # 注意力機制
        self.attention_dim = num_filters * len(filter_sizes)
        self.attention = nn.Linear(self.attention_dim, 1)
        
        # Dropout 和分類層
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.attention_dim, num_labels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, return_attention=False):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # 多尺度卷積
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, conv_len)
            conv_outputs.append(conv_out)
        
        # 全局最大池化
        pooled_outputs = []
        for conv_out in conv_outputs:
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch_size, num_filters, 1)
            pooled = pooled.squeeze(2)  # (batch_size, num_filters)
            pooled_outputs.append(pooled)
        
        # 連接所有特徵
        features = torch.cat(pooled_outputs, dim=1)  # (batch_size, attention_dim)
        
        # 注意力機制
        # 為了計算注意力權重，我們需要重新整理特徵
        batch_size = features.size(0)
        attention_input = features.unsqueeze(1)  # (batch_size, 1, attention_dim)
        attention_weights = torch.softmax(self.attention(attention_input), dim=1)  # (batch_size, 1, 1)
        
        # 應用注意力
        attended_features = attention_weights * features.unsqueeze(1)  # (batch_size, 1, attention_dim)
        attended_features = attended_features.squeeze(1)  # (batch_size, attention_dim)
        
        # 分類
        attended_features = self.dropout(attended_features)
        output = self.fc(attended_features)  # (batch_size, num_labels)
        output = self.sigmoid(output)
        
        if return_attention:
            return output, attention_weights.mean().item()
        return output

# 5. 自定義數據集
class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# 6. 評估指標函數
def hamming_score(y_true, y_pred):
    return np.mean([
        np.logical_and(t, p).sum() / np.logical_or(t, p).sum() if np.logical_or(t, p).sum() != 0 else 1.0
        for t, p in zip(y_true, y_pred)
    ])

def compute_metrics(y_true, y_pred, inference_time, num_samples, gpu_mem_usage, attention_weight):
    """計算評估指標"""
    
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    subset_acc = accuracy_score(y_true, y_pred)
    hamming = hamming_score(y_true, y_pred)
    
    return {
        "eval_subset_accuracy": subset_acc,
        "eval_hamming_score": hamming,
        "eval_precision_micro": precision_micro,
        "eval_recall_micro": recall_micro,
        "eval_f1_micro": f1_micro,
        "eval_precision_macro": precision_macro,
        "eval_recall_macro": recall_macro,
        "eval_f1_macro": f1_macro,
        "avg_inference_time": inference_time,
        "inference_time_per_sample": inference_time / num_samples,
        "samples_per_second": num_samples / inference_time,
        "gpu_mem_usage": gpu_mem_usage,
        "avg_attention_weight": attention_weight
    }

# 7. 訓練函數
def train_model(model, train_loader, val_loader, num_epochs=15, lr=0.001, pos_weights=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 設定損失函數和優化器
    if pos_weights is not None:
        pos_weights = torch.FloatTensor(pos_weights).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        criterion = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            if pos_weights is not None:
                # 使用 logits 計算損失
                outputs_logits = torch.log(outputs / (1 - outputs + 1e-8))
                loss = criterion(outputs_logits, batch_y)
            else:
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # 驗證
        val_loss = evaluate_model(model, val_loader, criterion, device, pos_weights is not None)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

def evaluate_model(model, data_loader, criterion, device, use_logits=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            if use_logits:
                outputs_logits = torch.log(outputs / (1 - outputs + 1e-8))
                loss = criterion(outputs_logits, batch_y)
            else:
                loss = criterion(outputs, batch_y)
                
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(data_loader)

# 8. 交叉驗證主流程
print("構建詞彙表...")
vocab = build_vocab(data['processed_text'].tolist())
print(f"詞彙表大小: {len(vocab)}")

print("載入詞向量...")
embeddings = load_word_embeddings(vocab, embedding_dim=100)

print("轉換文本為序列...")
sequences = texts_to_sequences(data['processed_text'].tolist(), vocab, max_len=200)

print("開始 10 折交叉驗證...")

kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []
best_model = None
best_f1 = 0.0

for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
    print(f"\n{'='*40}")
    print(f"Fold {fold+1}/10 數據量統計")
    print(f"{'='*40}")
    print(f"原始完整數據集: {len(data)} 樣本")
    print(f"當前fold訓練集: {len(train_idx)} 樣本")
    print(f"當前fold驗證集: {len(val_idx)} 樣本")
    
    # 清理之前的 GPU 記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 分割資料
    train_sequences = sequences[train_idx]
    train_labels = labels[train_idx]
    
    val_sequences = sequences[val_idx]
    val_labels = labels[val_idx]
    
    # 顯示 fold 訓練數據分佈
    analyze_data_distribution(train_labels, f"Fold {fold+1} 訓練數據分佈")
    
    # 應用 SMOTE 重新採樣
    print("開始 SMOTE 重新採樣...")
    train_sequences_resampled, train_labels_resampled, smote_applied = apply_smote_to_sequences(
        train_sequences, train_labels, random_state=42+fold
    )
    
    # 顯示 SMOTE 後的數據分佈
    analyze_data_distribution(train_labels_resampled, f"Fold {fold+1} SMOTE 重新採樣後數據分佈")
    
    # 計算類別權重
    pos_weights = []
    for i in range(3):
        n_pos = train_labels_resampled[:, i].sum()
        n_neg = len(train_labels_resampled) - n_pos
        pos_weights.append(n_neg / (n_pos + 1e-6))
    
    print(f"類別權重: {[f'{w:.2f}' for w in pos_weights]}")
    
    # 創建數據載入器
    train_dataset = TextDataset(train_sequences_resampled, train_labels_resampled)
    val_dataset = TextDataset(val_sequences, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    model = CNNAttentionClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        num_filters=128,  # 每個卷積核的數量
        filter_sizes=[3, 4, 5],  # 不同的卷積核大小
        num_labels=3,
        embeddings=embeddings,
        dropout=0.3
    )
    
    # 訓練模型
    print("訓練 CNN+Attention 模型...")
    start_train_time = time.perf_counter()
    train_model(model, train_loader, val_loader, num_epochs=15, lr=0.001, pos_weights=pos_weights)
    train_time = time.perf_counter() - start_train_time
    print(f"訓練時間: {train_time:.3f}s")
    
    # 評估模型
    print("進行模型評估...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_attention_weights = []
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            outputs, attention_weight = model(batch_x, return_attention=True)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int)
            all_predictions.extend(predictions)
            all_attention_weights.append(attention_weight)
    
    inference_time = time.perf_counter() - start_time
    
    # GPU 記憶體使用量
    gpu_mem_usage = 0
    if torch.cuda.is_available():
        gpu_mem_usage = torch.cuda.max_memory_allocated() / 1e9
    
    # 平均注意力權重
    avg_attention_weight = np.mean(all_attention_weights)
    
    # 計算評估指標
    eval_results = compute_metrics(
        val_labels, np.array(all_predictions),
        inference_time, len(val_labels), 
        gpu_mem_usage, avg_attention_weight
    )
    
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
        if best_model is not None:
            del best_model
        best_model = model
        print(f"新的最佳模型! Macro F1: {best_f1:.3f}")
    else:
        del model
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 9. 交叉驗證結果匯總
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

# 10. 保存最佳模型
if best_model is not None:
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'vocab': vocab,
        'embeddings': embeddings
    }, "./cnn_attention_3label_model.pth")
    print(f"\n最佳模型已保存! (Macro F1: {best_f1:.3f})")
else:
    print("\n警告: 沒有找到可保存的模型")

# 11. 視覺化分析
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
plt.title('Memory vs Performance (CNN+Attention)')

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
plt.title('Inference Time vs Performance (CNN+Attention)')

# 子圖3: 各fold效能比較
plt.subplot(1, 3, 3)
folds = range(1, len(fold_results) + 1)
plt.plot(folds, [r['eval_f1_macro'] for r in fold_results], 'bo-', label='Macro F1')
plt.plot(folds, [r['eval_f1_micro'] for r in fold_results], 'ro-', label='Micro F1')
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('Performance Across Folds (CNN+Attention)')
plt.legend()

plt.tight_layout()
plt.savefig('./cnn_attention_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 12. 可解釋性報告
print("\n可解釋性分析:")
avg_attention = avg_metrics['avg_attention_weight']
print(f"平均注意力權重: {avg_attention:.6f}")
print("注意力權重較高表示模型對輸入文本的關注度較高")

# 添加詳細的注意力權重分析
attention_weights = [r.get('avg_attention_weight', 0) for r in fold_results]
print(f"注意力權重標準差: {np.std(attention_weights):.6f}")
print(f"注意力權重範圍: {np.min(attention_weights):.6f} - {np.max(attention_weights):.6f}")

print("\nCNN+Attention 模型特點:")
print("- 使用多尺度卷積神經網絡捕捉局部特徵")
print("- 集成注意力機制提高模型可解釋性")
print("- 通過全局最大池化提取關鍵特徵")
print("- 支持 GPU 加速訓練和推理")
print("- 適合處理文本分類任務")