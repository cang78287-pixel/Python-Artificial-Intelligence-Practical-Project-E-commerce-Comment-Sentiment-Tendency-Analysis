import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

# 1. 读取数据
df = pd.read_csv("chnsenticorp_cleaned.csv", encoding="utf-8")
print("======= 默认参数版 =======")
print("\n数据形状：", df.shape)

X_text = df["clean_text"].astype(str)
y = df["label"].astype(int)

# 2. TF-IDF 特征提取
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X_text)
with open("tfidf.pkl", "wb") as f:
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
        "宏平均F1": round(f1_macro, 4)
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
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# ------------------------------------------------------------------------------
# 6. 调参版
# ------------------------------------------------------------------------------
print("\n======= 调参版（GridSearchCV）=======")
print("\n数据形状：", df.shape)

# 朴素贝叶斯调参
print("\n朴素贝叶斯 调参")
nb = MultinomialNB()
nb_params = {"alpha": [0.1, 0.5, 1.0, 2.0]}
nb_grid = GridSearchCV(nb, nb_params, cv=5, scoring="accuracy")
nb_grid.fit(X_train, y_train)
nb_best = nb_grid.best_estimator_
print(f"最优参数：{nb_grid.best_params_}")

# 逻辑回归调参
print("\n逻辑回归 调参")
lr = LogisticRegression(max_iter=1000)
lr_params = {"C": [0.1, 1, 10]}
lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring="accuracy")
lr_grid.fit(X_train, y_train)
lr_best = lr_grid.best_estimator_
print(f"最优参数：{lr_grid.best_params_}")

# 调参后模型
tuned_models = {
    "朴素贝叶斯_tuned": nb_best,
    "逻辑回归_tuned": lr_best
}

# 7. 训练调参后模型
print("\n训练调参后模型：")
for name, model in tuned_models.items():
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
        "宏平均F1": round(f1_macro, 4)
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
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# 8. 输出最终表格
pd.DataFrame(model_report).to_csv("model_report.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(model_summary).to_csv("model_summary.csv", index=False, encoding="utf-8-sig")

print("\n完成")