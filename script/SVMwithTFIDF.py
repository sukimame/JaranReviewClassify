import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/pn.tsv', sep="\t", header=None, names=['num', 'label', 'text', 'judges', 'usage'])

"""
# 二値分類のためにニュートラルを除外
df_bin = df[df["label"] != 0]
df_bin["label"] = (df_bin["label"] == 1).astype(int)

train_df = df_bin[df_bin['usage'] == 'train']
dev_df   = df_bin[df_bin['usage'] == 'dev']
test_df  = df_bin[df_bin['usage'] == 'test']
"""

df_train = df[df['usage'] == 'train']
df_dev   = df[df['usage'] == 'dev']
df_test  = df[df['usage'] == 'test']

vectorizer = TfidfVectorizer()
train_X = vectorizer.fit_transform(df_train["text"].values)
dev_X   = vectorizer.transform(df_dev["text"].values)
test_X  = vectorizer.transform(df_test["text"].values)

train_y =  df_train['label'].values
dev_y   = df_dev['label'].values
test_y  = df_test['label'].values

"""
clf = Pipeline([
    ("svm", LinearSVC(
        C=0.05,
        max_iter=10000,
        class_weight="balanced",
    ))
])
"""

clf = Pipeline([
    ("svm", LinearSVC(
        C=0.05,
        max_iter=10000,
        class_weight="balanced",
        multi_class="ovr"
    ))
])

clf.fit(train_X, train_y)

# トレーニングデータに対する精度
pred_train = clf.predict(train_X)
accuracy_train = accuracy_score(train_y, pred_train)
print('train： %.2f' % accuracy_train)
print("dev:", accuracy_score(dev_y, clf.predict(dev_X)))
#print("test:", accuracy_score(test_y, clf.predict(test_X))) #検証用データ

y_pred = clf.predict(dev_X)
# 誤分類されたデータのインデックス
wrong_idx = np.where(y_pred != dev_y)[0]

print("誤分類数:", len(wrong_idx))

np.save('tfidf_widx.npy', wrong_idx)
np.save('tfidf_dev_pred.npy', y_pred)

def print_wrong_examples(df_dev, wrong_idx):
    for idx in wrong_idx:
        true_label = dev_y[idx]
        predicted_label = y_pred[idx]
        text = df_dev.iloc[idx]['text']
        print(f"Index: {idx}, True Label: {true_label}, Predicted Label: {predicted_label}")
        print(f"Text: {text}\n")

# print_wrong_examples(df_dev, wrong_idx)