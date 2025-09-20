import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import Dataset, DataLoader
from evaluate import load as load_metric  # datasets 1.18.0 版本開始，load_metric 被移動到了 evaluate 庫
from sklearn.model_selection import train_test_split, KFold


# 讀取訓練數據的 CSV 檔案
data = pd.read_csv('../2_label_totalData.csv', encoding='utf-8')
print(data.head())

# 初始化 BERT 分詞器
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 定義分詞函數
def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=200)

# 對整個數據集進行分詞
tokenized_inputs = tokenize_texts(data['text'].tolist())

# 準備數據進行過採樣
input_ids = tokenized_inputs['input_ids'].numpy()  # 將 input_ids 提取出來作為過採樣特徵
labels = data['label'].values  # 提取標籤

# 分割數據集為訓練集與測試集
train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    tokenized_inputs['input_ids'].numpy(), labels, test_size=0.2, random_state=42)

train_masks, test_masks = train_test_split(
    tokenized_inputs['attention_mask'].numpy(), test_size=0.2, random_state=42)

# 對訓練集進行 SMOTE 過採樣
undersampler = RandomUnderSampler(random_state=42)
train_inputs_resampled, train_labels_resampled = undersampler.fit_resample(train_inputs, train_labels)
train_masks_resampled, _ = undersampler.fit_resample(train_masks, train_labels)


# 在訓練集上進行 10-fold 交叉驗證
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(train_inputs_resampled)):
    # 分割每一折的訓練與驗證資料
    train_fold_inputs, val_fold_inputs = train_inputs_resampled[train_index], train_inputs_resampled[val_index]
    train_fold_labels, val_fold_labels = train_labels_resampled[train_index], train_labels_resampled[val_index]
    train_fold_masks, val_fold_masks = train_masks_resampled[train_index], train_masks_resampled[val_index]

    

# 創建自定義 Dataset 類
class CustomDataset(Dataset):
    def __init__(self, input_ids, labels, attention_mask):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

# 創建訓練和測試數據集
trainDataset = CustomDataset(train_inputs_resampled, train_labels_resampled, train_masks_resampled)
testDataset = CustomDataset(test_inputs, test_labels, test_masks)

# 創建 DataLoader
train_loader = DataLoader(trainDataset, batch_size=2, shuffle=True)
test_loader = DataLoader(testDataset, batch_size=2, shuffle=True)

# 加載評估指標
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)['accuracy']
    precision = precision_metric.compute(predictions=preds, references=labels, average='weighted')['precision']
    f1 = f1_metric.compute(predictions=preds, references=labels, average='weighted')['f1']
    
    return {"accuracy": accuracy, "precision": precision, "f1": f1}

# 設置訓練參數
training_args = TrainingArguments(
    output_dir='./results',          # 儲存模型和結果的目錄
    evaluation_strategy="epoch",     # 每個訓練週期結束時進行評估
    learning_rate=2e-5,              # 設定學習率
    per_device_train_batch_size=2,  # 每個設備 (如 GPU) 的訓練批次大小
    per_device_eval_batch_size=2,
    num_train_epochs=3,              # 訓練週期數
    weight_decay=0.01,               # 權重衰減率，防止過度擬合
)

# 定義訓練器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trainDataset,       # 訓練數據集
    eval_dataset=testDataset,           # 驗證數據集
    compute_metrics=compute_metrics,  # 傳入計算指標的函數
)

# 進行模型微調
trainer.train()

train_metrics = trainer.evaluate(eval_dataset=trainDataset)  # 評估模型表現
store_fine_tuned = "bert-base-chinese"


train_results = []
eval_accuracy_results = []
eval_precision_results = []
eval_f1_results = []
eval_runtime_results = []

for i in range(3):
    print(f"Training iteration {i+1}")
    trainer.train()  # 訓練模型

    # 計算訓練集的表現
    train_metrics = trainer.evaluate(eval_dataset=trainDataset)
    print(f"Training_result (Iteration {i+1}):")
    for key, value in train_metrics.items():
        print(f"{key}: {value}")
    
    # 儲存訓練準確率
    train_results.append(train_metrics.get("eval_accuracy", 0))  # 儲存訓練準確率

    # 儲存模型和分詞器
    trainer.save_model(f"{store_fine_tuned}_iteration_{i+1}")
    tokenizer.save_pretrained(f"{store_fine_tuned}_iteration_{i+1}")

    # 計算驗證集的表現
    eval_metrics = trainer.evaluate(eval_dataset=testDataset)
    print(f"Evaluation_result (Iteration {i+1}):")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")

    # 儲存各指標
    if "eval_accuracy" in eval_metrics:
        eval_accuracy_results.append(eval_metrics["eval_accuracy"])
    if "eval_precision" in eval_metrics:
        eval_precision_results.append(eval_metrics["eval_precision"])
    if "eval_f1" in eval_metrics:
        eval_f1_results.append(eval_metrics["eval_f1"])   
    if "eval_runtime" in eval_metrics:
        eval_runtime_results.append(eval_metrics["eval_runtime"])

# 計算平均值和標準差
eval_accuracy_mean = np.mean(eval_accuracy_results)
eval_accuracy_std = np.std(eval_accuracy_results)
eval_precision_mean = np.mean(eval_precision_results)
eval_precision_std = np.std(eval_precision_results)
eval_f1_mean = np.mean(eval_f1_results)
eval_f1_std = np.std(eval_f1_results)
eval_runtime_mean = np.mean(eval_runtime_results)

print(f"Evaluation Accuracy 的平均值與標準差: {eval_accuracy_mean:.3f} ± {eval_accuracy_std:.3f}")
print(f"Evaluation Precision 的平均值與標準差: {eval_precision_mean:.3f} ± {eval_precision_std:.3f}")
print(f"Evaluation F1 的平均值與標準差: {eval_f1_mean:.3f} ± {eval_f1_std:.3f}")
print(f"Evaluation Runtime 的平均值與標準差: {eval_runtime_mean:.2f}")

# 從驗證集生成預測
predictions, labels, _ = trainer.predict(testDataset)

# 轉換預測為分類標籤, axis 表示在每個樣本的預測向量中選擇最大值的索引。這一步將預測的連續分數轉換為具體的分類標籤
predicted_labels = np.argmax(predictions, axis=1)

# 計算混淆矩陣
cm = confusion_matrix(labels, predicted_labels)

# 顯示混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Training Metrics")
plt.show()

print(f"測試集大小: {len(testDataset)}")

cm = confusion_matrix(labels, predicted_labels)
print(f"混淆矩陣的數字加總: {cm.sum()}")