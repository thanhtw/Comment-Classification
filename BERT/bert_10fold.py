import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments, BertConfig
from datasets import Dataset, load_dataset  # 拆分訓練集、測試集
from evaluate import load as load_metric  # datasets 1.18.0 版本開始，load_metric 被移動到了 evaluate 庫
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, DataCollatorWithPadding, BertForSequenceClassification, TrainingArguments, Trainer  # BERT 微調、分類

# 匯入 csv，指定編碼為 utf-8
data = pd.read_csv('../2_label_totalData.csv', encoding= 'utf-8')

# 查看資料集前幾行，確認已正確讀入
print(data.head())

### Preprocess the dataset
def preprocess_function(examples):  # 將資料轉換為模型所需格式。examples: 待處理數據的字典或數據集。trunction: 截斷
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=200)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    # 確保 predictions 和 labels 是整數
    predictions = predictions.astype(int)
    labels = labels.astype(int)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision = precision_score(labels, predictions, average="weighted")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "precision": precision, "f1": f1}

# label to int, csv ouput is foat
label_to_int = {label: idx for idx, label in enumerate(data["label"].unique())}
data["label"] = data["label"].map(label_to_int)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)  # 劃分訓練集和測試集，80% 作為訓練集，20% 作為測試集

# 在訓練集上進行 10-fold 交叉驗證
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, val_index in kf.split(train_data):
    train_fold, val_fold = train_data.iloc[train_index], train_data.iloc[val_index]
    # 在這裡訓練模型並評估模型表現

# 將劃分後的數據保存為 csv
train_data.to_csv('train_set.csv', index=False)
test_data.to_csv('test_set.csv', index=False)

dataset = load_dataset('csv', data_files={'train': 'train_set.csv', 'test': 'test_set.csv'})  # 載入數據集csv
print(dataset)

model_id = "bert-base-chinese"  # 替換為你選擇的 BERT 模型 ID
store_fine_tuned = "./fine-tuned-BERT-chinese"
training_store = "./BERT_results"

config = BertConfig.from_pretrained(
    model_id,
    hidden_dropout_prob=0.2,        # Dropout 機率
    attention_probs_dropout_prob=0.1,  # 注意力 Dropout
    num_labels=2                    # 設定二分類
)
tokenizer = BertTokenizer.from_pretrained(model_id)
model = BertForSequenceClassification.from_pretrained(model_id, config=config)

# 定義一個函數來分詞數據集
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 劃分訓練集80、測試集20
trainDataset = tokenized_dataset['train']
testDataset = tokenized_dataset['test']

# 載入並準備好評估工具
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
f1_metric = load_metric("f1")

def convert_labels_to_int(example):
    example['label'] = int(example['label'])
    return example

tokenized_datasets = tokenized_datasets.map(convert_labels_to_int)

training_args = TrainingArguments(
    fp16=True,  # 啟用半精度訓練
    output_dir=training_store,       # 輸出模型和結果的目錄
    eval_strategy="epoch",           # 在每個訓練週期結束時進行評估
    save_strategy="epoch",           # 在每個訓練週期結束時儲存模型
    learning_rate=5e-5,              # 設定學習率
    per_device_train_batch_size=8,  # 每個設備 (如 GPU) 的訓練批次大小
    per_device_eval_batch_size=8,   # 每個設備的評估批次大小
    num_train_epochs=3,              # 訓練週期數
    weight_decay=0.01,               # 權重衰減率，防止過度擬合
    logging_dir='./logs',            # 訓練過程的日誌存放目錄
    logging_steps=10,                # 每 10 步記錄一次訓練資訊
    load_best_model_at_end=True,     # 在訓練結束時載入最佳模型
    metric_for_best_model="f1",      # 以 F1 分數來選擇最佳模型
    save_total_limit=2,  # 限制保存的檔案數量，避免磁碟空間不足
    gradient_accumulation_steps=4,  # 累積 4 步後進行梯度更新
)

# 定義訓練器
trainer = Trainer(
        model=model,                     # 要訓練的模型
        args=training_args,              # 訓練參數
        train_dataset=trainDataset,     # 訓練數據集
        eval_dataset=testDataset,        # 驗證數據集
        compute_metrics=compute_metrics, # 評估指標計算函數
)

try:
    trainer.train()
except Exception as e:
    print(f"An error occurred: {e}")
    # 這裡可以添加更多的調試信息

train_results = []
eval_accuracy_results = []
eval_precision_results = []
eval_f1_results = []
eval_runtime_results = []

for i in range(3):
    print(f"Training iteration {i+1}")

    # 訓練模型
    trainer.train()
    torch.cuda.empty_cache()  # 清理顯存以避免 OOM

    # 訓練集表現評估
    train_metrics = trainer.evaluate(eval_dataset=trainDataset)
    print(f"Training results (Iteration {i+1}):")
    for key, value in train_metrics.items():
        print(f"{key}: {value}")

    # 儲存訓練集的準確率結果
    if "accuracy" in train_metrics:
        train_results.append(train_metrics["accuracy"])
    else:
        print("Warning: 'accuracy' not found in train_metrics.")


    # 驗證集或測試集表現評估
    eval_metrics = trainer.evaluate(eval_dataset=testDataset)
    print(f"Evaluation results (Iteration {i+1}):")
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

    # 儲存模型和分詞器
    model.save_pretrained(f'./saved_model_iteration_{i+1}')
    tokenizer.save_pretrained(f'./saved_model_iteration_{i+1}')
    torch.cuda.empty_cache()  # 再次清理顯存


# 保存model和分詞器
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
     
# 計算平均值和標準差
eval_accuracy_mean = np.mean(eval_accuracy_results)
eval_accuracy_std = np.std(eval_accuracy_results)
eval_precision_mean = np.mean(eval_precision_results)
eval_precision_std = np.std(eval_precision_results)
eval_f1_mean = np.mean(eval_f1_results)
eval_f1_std = np.std(eval_f1_results)
eval_runtime_mean = np.mean(eval_runtime_results)
eval_runtime_std = np.std(eval_runtime_results)

print(f"Evaluation Accuracy 的平均值與標準差: {eval_accuracy_mean:.3f} ± {eval_accuracy_std:.3f}")
print(f"Evaluation Precision 的平均值與標準差: {eval_precision_mean:.3f} ± {eval_precision_std:.3f}")
print(f"Evaluation F1 的平均值與標準差: {eval_f1_mean:.3f} ± {eval_f1_std:.3f}")
print(f"Evaluation Runtime 的平均值與標準差: {eval_runtime_mean:.3f} ± {eval_runtime_std:.3f}")



# 從驗證集生成預測
predictions, labels, _ = trainer.predict(testDataset)

# 轉換預測為分類標籤, axis 表示在每個樣本的預測向量中選擇最大值的索引。這一步將預測的連續分數轉換為具體的分類標籤
predictions = torch.nn.functional.softmax(predictions, dim=-1)
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