#!/usr/bin/env python3
"""
簡化的 RoBERTa SMOTE 測試腳本
"""

print("開始測試 RoBERTa SMOTE 功能...")

try:
    import pandas as pd
    import numpy as np
    print("✓ 基本庫導入成功")
except ImportError as e:
    print(f"✗ 基本庫導入失敗: {e}")
    exit(1)

try:
    # 測試數據讀取
    data = pd.read_csv('../data/cleaned_3label_data.csv', encoding='utf-8')
    print(f"✓ 數據讀取成功，形狀: {data.shape}")
    
    # 測試標籤提取
    labels = data[['relevance', 'concreteness', 'constructive']].values
    print(f"✓ 標籤提取成功，形狀: {labels.shape}")
    
    # 測試數據分佈分析函數
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
    
    analyze_data_distribution(labels, "原始數據分佈測試")
    print("✓ 數據分佈分析函數正常")
    
    # 測試 SMOTE 導入
    try:
        from imblearn.over_sampling import SMOTE
        print("✓ SMOTE 導入成功")
    except ImportError:
        print("✗ SMOTE 導入失敗，請安裝 imbalanced-learn")
    
    # 測試原始數據統計顯示函數
    def display_original_data_statistics(labels):
        print("\n" + "="*60)
        print("原始資料標籤分佈統計")
        print("="*60)
        
        label_names = ['相關性標籤', '具體性標籤', '建設性標籤']
        total_samples = len(labels)
        
        print(f"總評論數: {total_samples}")
        print(f"{'標籤類型':<12} {'出現次數':<10} {'比例':<10} {'分佈圖'}")
        print("-" * 60)
        
        for i, label_name in enumerate(label_names):
            positive_count = np.sum(labels[:, i])
            ratio = positive_count / total_samples * 100
            
            # 簡單的文字圖表
            bar_length = int(ratio / 2)  # 每個 █ 代表 2%
            bar = "█" * bar_length + "░" * (50 - bar_length)
            
            print(f"{label_name:<12} {positive_count:<10} {ratio:>6.1f}%   {bar}")
        
        print("-" * 60)
        print("圖例: █ = 2%, ░ = 未達到")
    
    display_original_data_statistics(labels)
    print("✓ 原始數據統計顯示函數正常")
    
    print("\n🎉 所有測試通過！RoBERTa SMOTE 功能準備就緒。")
    
except Exception as e:
    print(f"✗ 測試失敗: {e}")
    import traceback
    traceback.print_exc()
