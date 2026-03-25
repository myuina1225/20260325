import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# さまざまな機械学習モデル群
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def evaluate_model(model, X_train, y_train, X_eval, y_eval, label):
    """
    指定されたモデルを学習・評価する関数
    """
    # モデルの訓練
    model.fit(X_train, y_train)
    
    # 評価
    predictions = model.predict(X_eval)
    
    # ROC-AUC用の離職確率 (predict_proba) を取得
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_eval)[:, 1]
    elif hasattr(model, "decision_function"):
        probas = model.decision_function(X_eval)
    else:
        # 確率が出力できないモデルのフォールバック
        probas = predictions
        
    acc = accuracy_score(y_eval, predictions)
    prec = precision_score(y_eval, predictions, zero_division=0)
    rec = recall_score(y_eval, predictions, zero_division=0)
    f1 = f1_score(y_eval, predictions, zero_division=0)
    auc = roc_auc_score(y_eval, probas)
    
    print(f"[{label}]")
    print(f"  - Accuracy : {acc:.4f}")
    print(f"  - Precision: {prec:.4f}")
    print(f"  - Recall   : {rec:.4f}")
    print(f"  - F1 Score : {f1:.4f}")
    print(f"  - ROC-AUC  : {auc:.4f}\n")
    
    return auc

def main():
    # グローバルなシード値の固定
    set_seed(42)
    
    data_path = '/home/mizuashi/educ_ml/gci/dataset/data2.csv'
    output_dir = '/home/mizuashi/educ_ml/gci/code'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # データの読み込み
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} が見つかりません。")
        return
    
    # Attritionの Yes/No を 1/0 に変換
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    
    # 訓練データと評価データに8:2で分割 (不均衡データに対応するため stratify を指定)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Attrition'])
    
    # csvとして保存
    train_path = os.path.join(output_dir, 'train.csv')
    eval_path = os.path.join(output_dir, 'eval.csv')
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    
    # 機械学習用のデータ整形 (目的変数の分離)
    # 今回は「全部入り」に特化するため、カラム全体を含めている状態です。
    X_train = train_df.drop('Attrition', axis=1)
    y_train = train_df['Attrition']
    
    X_eval = eval_df.drop('Attrition', axis=1)
    y_eval = eval_df['Attrition']
    
    print("=== 全ての特徴量 (全部入り) を用いたモデル比較実験 ===\n")
    print(f"使用特徴量数: {X_train.shape[1]}")
    print(f"使用特徴量: {', '.join(X_train.columns)}\n")
    
    # 比較する機械学習モデルの定義
    # 前回のユーザー指示を尊重し RandomForest の n_estimators=50 を採用
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', max_depth=10),
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "SVM (RBF Kernel)": SVC(probability=True, random_state=42, class_weight='balanced')
    }
    
    results = {}
    for label, model in models.items():
        try:
            auc_score = evaluate_model(model, X_train, y_train, X_eval, y_eval, label)
            results[label] = auc_score
        except Exception as e:
            print(f"[{label}] モデルの評価中にエラーが発生しました: {e}\n")
        
    print("=== 最終的な ROC-AUC ランキング 20260326===")
    # ROC-AUC の高い順にソートして表示
    for label, auc in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{label}: {auc:.4f}")

if __name__ == '__main__':
    main()
