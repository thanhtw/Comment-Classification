import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from evaluate import load as load_metric

# 讀取數據
data = pd.read_csv('../2_label_totalData.csv', encoding='utf-8')

# 初始化 BERT 分詞器與模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 分詞函數
def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='np', max_length=200)

# 對整個數據集進行分詞
tokenized_inputs = tokenize_texts(data['text'].tolist())
input_ids = tokenized_inputs['input_ids']
labels = data['label'].values

# K-Fold 交叉驗證
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 儲存混淆矩陣
conf_matrices = []

# 定義 Dataset
class CustomDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

# 設置訓練參數
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 設定評估指標
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

# 10-fold 訓練與測試
for fold, (train_index, val_index) in enumerate(kf.split(input_ids)):
    print(f"----- Fold {fold + 1} -----")

    train_fold_inputs, val_fold_inputs = input_ids[train_index], input_ids[val_index]
    train_fold_labels, val_fold_labels = labels[train_index], labels[val_index]

    # **對訓練集執行 SMOTE**
    smote = SMOTE(random_state=42)
    train_fold_inputs_resampled, train_fold_labels_resampled = smote.fit_resample(train_fold_inputs, train_fold_labels)

    trainDataset = CustomDataset(train_fold_inputs_resampled, train_fold_labels_resampled)
    valDataset = CustomDataset(val_fold_inputs, val_fold_labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainDataset,
        eval_dataset=valDataset,
        compute_metrics=compute_metrics,
    )

    # 訓練模型
    trainer.train()

    # 取得預測結果
    predictions, labels, _ = trainer.predict(valDataset)
    predicted_labels = np.argmax(predictions, axis=1)

    # 計算混淆矩陣
    cm = confusion_matrix(labels, predicted_labels)
    conf_matrices.append(cm)

# 計算 10-fold 平均混淆矩陣
avg_conf_matrix = np.mean(conf_matrices, axis=0)

# 顯示最終平均混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=avg_conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title("10-Fold Average Confusion Matrix")
plt.show()

print("10-Fold 平均混淆矩陣數據：")
print(avg_conf_matrix)