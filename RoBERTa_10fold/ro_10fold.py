#此程式碼有流程順序錯誤 尚未修改

import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset, DataLoader
from evaluate import load as load_metric  # datasets 1.18.0 版本開始，load_metric 被移動到了 evaluate 庫
from sklearn.model_selection import train_test_split, KFold


# 讀取訓練數據的 CSV 檔案
data = pd.read_csv('../2_label_totalData.csv', encoding='utf-8')
print(data.head())

# 初始化 BERT 分詞器
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
model = RobertaForSequenceClassification.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=2)

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

# 在訓練集上進行 10-fold 交叉驗證
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(train_inputs)):
    # 分割每一折的訓練與驗證資料
    train_fold_inputs, val_fold_inputs = train_inputs[train_index], train_inputs[val_index]
    train_fold_labels, val_fold_labels = train_labels[train_index], train_labels[val_index]
    train_fold_masks, val_fold_masks = train_masks[train_index], train_masks[val_index]

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
trainDataset = CustomDataset(train_inputs, train_labels, train_masks)
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
plt.title("RoBERTa Training Metrics")
plt.show()

print(f"測試集大小: {len(testDataset)}")

cm = confusion_matrix(labels, predicted_labels)
print(f"混淆矩陣的數字加總: {cm.sum()}")





# import pandas as pd
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
# from datasets import Dataset, load_dataset  # 拆分訓練集、測試集
# from evaluate import load as load_metric  # datasets 1.18.0 版本開始，load_metric 被移動到了 evaluate 庫
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.model_selection import train_test_split, KFold
# from imblearn.over_sampling import SMOTE
# from transformers import AutoTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification, TrainingArguments, Trainer

# # 釋放不必要的記憶體 , 在訓練過程中，確保不必要的變量不再佔用顯存。使用 torch.cuda.empty_cache() 來釋放顯存。
# torch.cuda.empty_cache()

# # 匯入 csv，指定編碼為 utf-8
# data = pd.read_csv('2_label_totalData.csv', encoding= 'utf-8')

# # 查看資料集前幾行，確認已正確讀入
# print(data.head())

# ### Preprocess the dataset
# def preprocess_function(examples):  # 將資料轉換為模型所需格式。examples: 待處理數據的字典或數據集。trunction: 截斷
#     return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=120)

# # 計算指標函數，評估過程中的性能指標，包括準確率（accuracy）、精確率（precision）和 F1 分數（F1 score）
# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=1)

#     ## 呼叫計算在這裡
#     # 使用預設的accuracy_metric計算準確率
#     accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]

#     # 使用預設的precision_metric計算精確率，average="weighted"是指對不同類別的結果進行加權平均，以考慮樣本數量的不同
#     precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"]

#     # F1 score = 精確率（Precision）和召回率（Recall）的調和平均數
#     f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
#     return {"accuracy": accuracy, "precision": precision, "f1": f1}

# # label to int, csv ouput is foat
# label_to_int = {label: int for int, label in enumerate(data["label"].unique())}
# data["label"] = data["label"].map(label_to_int)

# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)  # 劃分訓練集和測試集，80% 作為訓練集，20% 作為測試集

# # 在訓練集上進行 10-fold 交叉驗證
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# for train_index, val_index in kf.split(train_data):
#     train_fold, val_fold = train_data.iloc[train_index], train_data.iloc[val_index]
#     # 在這裡訓練模型並評估模型表現

# # 將劃分後的數據保存為 csv
# train_data.to_csv('train_set.csv', index=False)
# test_data.to_csv('test_set.csv', index=False)

# dataset = load_dataset('csv', data_files={'train': 'train_set.csv', 'test': 'test_set.csv'})  # 載入數據集csv
# print(dataset)

# # 初始化 BERT 分詞器
# tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

# # 定義一個函數來分詞數據集
# def tokenize_function(examples):
#     return tokenizer(examples['text'], padding="max_length", truncation=True)

# # 對數據集中的 'text' 欄位應用分詞器
# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# # 查看分詞後的數據集結構
# print(tokenized_datasets)

# model_id = "hfl/chinese-roberta-wwm-ext"
# store_fine_tuned = "./fine-tuned-Robert-chinese"
# training_store ="./Robert_results"

# # 初始化 BERT 分詞器
# model = RobertaForSequenceClassification.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=2)

# # 將數據集應用分詞函數
# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # 劃分訓練集80、測試集20
# trainDataset = tokenized_dataset['train']
# testDataset = tokenized_dataset['test']

# # 載入並準備好評估工具
# accuracy_metric = load_metric("accuracy")
# precision_metric = load_metric("precision")
# f1_metric = load_metric("f1")

# def convert_labels_to_int(example):
#     # 先轉換為浮點數，再轉換為整數，確保處理浮點標籤
#     example['label'] = int(float(example['label']))
#     return example

# tokenized_datasets = tokenized_datasets.map(convert_labels_to_int)

# # Training Start

# training_args = TrainingArguments(
#         output_dir=training_store,       # 輸出模型和結果的目錄
#         eval_strategy="epoch",           # 在每個訓練週期結束時進行評估
#         save_strategy="epoch",           # 在每個訓練週期結束時儲存模型
#         learning_rate=2e-5,              # 設定學習率
#         per_device_train_batch_size=8,  # 每個設備 (如 GPU) 的訓練批次大小
#         per_device_eval_batch_size=8,   # 每個設備的評估批次大小
#         num_train_epochs=3,              # 訓練週期數
#         weight_decay=0.01,               # 權重衰減率，防止過度擬合
#         logging_dir='./logs',            # 訓練過程的日誌存放目錄
#         logging_steps=10,                # 每 10 步記錄一次訓練資訊
#         load_best_model_at_end=True,     # 在訓練結束時載入最佳模型
#         metric_for_best_model="f1",      # 以 F1 分數來選擇最佳模型
# )

# # 定義訓練器
# trainer = Trainer(
#         model=model,                     # 要訓練的模型
#         args=training_args,              # 訓練參數
#         train_dataset=trainDataset,     # 訓練數據集
#         eval_dataset=testDataset,        # 驗證數據集
#         compute_metrics=compute_metrics, # 評估指標計算函數
# )

# try:
#     trainer.train()
# except Exception as e:
#     print(f"An error occurred: {e}")
#     # 這裡可以添加更多的調試信息

# train_metrics = trainer.evaluate(eval_dataset=trainDataset)  # 評估模型表現

# train_results = []
# eval_accuracy_results = []
# eval_precision_results = []
# eval_f1_results = []
# eval_runtime_results = []

# for i in range(3):
#     print(f"Training iteration {i+1}")
#     trainer.train()  # 訓練模型

#     # 計算訓練集的表現
#     train_metrics = trainer.evaluate(eval_dataset=trainDataset)
#     eval_metrics = trainer.evaluate(eval_dataset=testDataset)

#     print(f"Training_result (Iteration {i+1}):")
#     for key, value in train_metrics.items():
#         print(f"{key}: {value}")
    
#     # 假設 "eval_accuracy" 是訓練結果中的準確率指標名稱，請替換為正確的名稱
#     if "eval_accuracy" in train_metrics:
#         train_results.append(train_metrics["eval_accuracy"])
#     else:
#         print("Warning: 'eval_accuracy' not found in train_metrics.")

#     # 儲存模型和分詞器
#     trainer.save_model(f"{store_fine_tuned}_iteration_{i+1}")
#     tokenizer.save_pretrained(f"{store_fine_tuned}_iteration_{i+1}")

#     eval_metrics = trainer.evaluate(eval_dataset=testDataset)

#     print(f"Evaluation_result (Iteration {i+1}):")
#     for key, value in eval_metrics.items():
#         print(f"{key}: {value}")

#     # 儲存各指標
#     if "eval_accuracy" in eval_metrics:
#         eval_accuracy_results.append(eval_metrics["eval_accuracy"])
#     if "eval_precision" in eval_metrics:
#         eval_precision_results.append(eval_metrics["eval_precision"])
#     if "eval_f1" in eval_metrics:
#         eval_f1_results.append(eval_metrics["eval_f1"])
#     if "eval_runtime" in eval_metrics:
#         eval_runtime_results.append(eval_metrics["eval_runtime"])

# # 計算平均值和標準差
# eval_accuracy_mean = np.mean(eval_accuracy_results)
# eval_accuracy_std = np.std(eval_accuracy_results)
# eval_precision_mean = np.mean(eval_precision_results)
# eval_precision_std = np.std(eval_precision_results)
# eval_f1_mean = np.mean(eval_f1_results)
# eval_f1_std = np.std(eval_f1_results)
# eval_runtime_mean = np.mean(eval_runtime_results)
# eval_runtime_std = np.std(eval_runtime_results)

# print(f"Evaluation Accuracy 的平均值與標準差: {eval_accuracy_mean:.3f} ± {eval_accuracy_std:.3f}")
# print(f"Evaluation Precision 的平均值與標準差: {eval_precision_mean:.3f} ± {eval_precision_std:.3f}")
# print(f"Evaluation F1 的平均值與標準差: {eval_f1_mean:.3f} ± {eval_f1_std:.3f}")
# print(f"Evaluation Runtime 的平均值與標準差: {eval_runtime_mean:.3f} ± {eval_runtime_std:.3f}")



# # 從驗證集生成預測
# predictions, labels, _ = trainer.predict(testDataset)

# # 轉換預測為分類標籤, axis 表示在每個樣本的預測向量中選擇最大值的索引。這一步將預測的連續分數轉換為具體的分類標籤
# predicted_labels = np.argmax(predictions, axis=1)

# # 計算混淆矩陣
# cm = confusion_matrix(labels, predicted_labels)

# # 顯示混淆矩陣
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Training Metrics")
# plt.show()

# print(f"測試集大小: {len(testDataset)}")

# cm = confusion_matrix(labels, predicted_labels)
# print(f"混淆矩陣的數字加總: {cm.sum()}")