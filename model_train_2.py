import pandas as pd
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# 锁定脚本所在目录，确保无论从哪运行都能找到同目录文件
DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 读取数据
df = pd.read_csv(os.path.join(DIR, "chnsenticorp_cleaned.csv"), encoding="utf-8")
print("======= 默认参数版 =======")
print("\n数据形状：", df.shape)

X_text = df["clean_text"].astype(str)
y = df["label"].astype(int)

# 2. TF-IDF 特征提取 (加入 bigram + 对数缩放，捕捉词组语义)
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),       # unigram + bigram，如 "很 不错" 不会被拆散
    sublinear_tf=True,        # 对数缩放，削弱长文本的权重优势
    max_df=0.9,               # 过滤出现于 90%+ 文档的极高频词
    min_df=2,                 # 过滤只出现 1 次的低频噪声词
)
X = tfidf.fit_transform(X_text)
with open(os.path.join(DIR, "tfidf.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

# 3. 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 默认参数模型
default_models = {
    "朴素贝叶斯": MultinomialNB(),
    "逻辑回归": LogisticRegression(max_iter=1000)
}

# 5. 先训练默认参数模型
model_report = []
model_summary = []

print("\n训练默认参数模型：")
for name, model in default_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    
    print(f"\n【{name}】")
    print(f"准确率: {acc:.4f} | F1(macro): {f1_macro:.4f}")
    report_text = classification_report(y_test, y_pred, target_names=["NEG", "POS"], digits=4)
    print(report_text)

    # 保存整体指标
    model_summary.append({
        "模型": name,
        "准确率": round(acc, 4),
        "宏平均F1": round(f1_macro, 4),
        "CV准确率": "-",
        "最优参数": "默认",
    })
    
    # 保存类别指标
    report = classification_report(y_test, y_pred, target_names=["NEG", "POS"], output_dict=True)
    for category, metrics in report.items():
        if category not in ["accuracy", "macro avg", "weighted avg"]:
            model_report.append({
                "模型": name,
                "类别": category,
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1-score": round(metrics["f1-score"], 4),
                "support": int(metrics["support"])
            })
    
    # 保存模型
    with open(os.path.join(DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)

# ------------------------------------------------------------------------------
# 6. 调参版
# ------------------------------------------------------------------------------
print("\n======= 调参版（GridSearchCV）=======")
print("\n数据形状：", df.shape)

# 朴素贝叶斯调参
print("\n朴素贝叶斯 调参")
nb = MultinomialNB()
nb_params = {
    "alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0],  # 细粒度搜索平滑系数
    "fit_prior": [True, False],                                   # 是否从数据学习先验概率
}
nb_grid = GridSearchCV(nb, nb_params, cv=5, scoring="accuracy")
nb_grid.fit(X_train, y_train)
nb_best = nb_grid.best_estimator_
print(f"最优参数：{nb_grid.best_params_}")
print(f"最优 CV 准确率：{nb_grid.best_score_:.4f}")

# 逻辑回归调参
print("\n逻辑回归 调参")
lr = LogisticRegression(max_iter=2000, random_state=42, solver="liblinear")
lr_params = {
    "C": [0.01, 0.1, 0.5, 1, 5, 10, 50],
    "penalty": ["l1", "l2"],
    "class_weight": [None, "balanced"],
}
lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring="accuracy", n_jobs=-1)
lr_grid.fit(X_train, y_train)
lr_best = lr_grid.best_estimator_
print(f"最优参数：{lr_grid.best_params_}")
print(f"最优 CV 准确率：{lr_grid.best_score_:.4f}")

# 调参后模型
tuned_models = {
    "朴素贝叶斯_tuned": nb_best,
    "逻辑回归_tuned": lr_best
}
# 保存对应的 grid 对象，用于提取 CV 准确率
tuned_grids = {
    "朴素贝叶斯_tuned": nb_grid,
    "逻辑回归_tuned": lr_grid,
}

# 7. 训练调参后模型
print("\n训练调参后模型：")
for name, model in tuned_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    grid = tuned_grids[name]
    cv_score = round(grid.best_score_, 4)
    best_params = str(grid.best_params_)
    
    print(f"\n【{name}】")
    print(f"准确率: {acc:.4f} | F1(macro): {f1_macro:.4f} | CV准确率: {cv_score:.4f}")
    report_text = classification_report(y_test, y_pred, target_names=["NEG", "POS"], digits=4)
    print(report_text)
    cm = confusion_matrix(y_test, y_pred)
    print(f"混淆矩阵:\n{cm}")

    # 保存整体指标
    model_summary.append({
        "模型": name,
        "准确率": round(acc, 4),
        "宏平均F1": round(f1_macro, 4),
        "CV准确率": cv_score,
        "最优参数": best_params,
    })
    
    # 保存类别指标
    report = classification_report(y_test, y_pred, target_names=["NEG", "POS"], output_dict=True)
    for category, metrics in report.items():
        if category not in ["accuracy", "macro avg", "weighted avg"]:
            model_report.append({
                "模型": name,
                "类别": category,
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1-score": round(metrics["f1-score"], 4),
                "support": int(metrics["support"])
            })
    
    # 保存模型
    with open(os.path.join(DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)

# 8. 输出最终表格
pd.DataFrame(model_report).to_csv(os.path.join(DIR, "model_report.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(model_summary).to_csv(os.path.join(DIR, "model_summary.csv"), index=False, encoding="utf-8-sig")

print("\n完成")