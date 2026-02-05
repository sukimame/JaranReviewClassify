import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from scipy import sparse
import pandas as pd

ylabel = np.load('data/ylabel.npz', allow_pickle=True)

train_X = sparse.load_npz('data/X_train.npz')
dev_X   = sparse.load_npz('data/X_dev.npz')
test_X  = sparse.load_npz('data/X_test.npz')

train_y =  ylabel['y_train']
dev_y = ylabel['y_dev']
test_y = ylabel['y_test']

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
"""

print(type(train_X))
print(getattr(train_X, "shape", None))

clf.fit(train_X, train_y)

# トレーニングデータに対する精度
pred_train = clf.predict(train_X)
accuracy_train = accuracy_score(train_y, pred_train)
print('トレーニングデータに対する正解率： %.2f' % accuracy_train)
print("dev:", accuracy_score(dev_y, clf.predict(dev_X)))
print("test:", accuracy_score(test_y, clf.predict(test_X)))

# 特徴量とラベル
X_test = sparse.load_npz("data/X_test.npz")
y = np.load("data/ylabel.npz")
y_test = y["y_test"]

# 元テキスト（testのみ）
df = pd.read_csv(
    "data/pn.tsv",
    sep="\t",
    header=None,
    names=["num", "label", "text", "judges", "usage"]
)

test_df = df[df["usage"] == "test"].reset_index(drop=True)

y_pred = clf.predict(X_test)

wrong_idx = np.where(y_pred != y_test)[0]

print("誤分類数:", len(wrong_idx))
print("最初の10件:", wrong_idx[:10])



for i in wrong_idx[:20]:
    print("|",  y_test[i], "|", y_pred[i], "|", test_df.loc[i, "text"], "|")

    scores = clf.decision_function(X_test)
confidence = np.abs(scores)

ambiguous_idx = np.argsort(confidence)[:10]

print("境界付近の文:")
for i in ambiguous_idx:
    print("----")
    print("score:", scores[i])
    print("text :", test_df.loc[i, "text"])
    print("gold :", y_test[i])
    print("pred :", y_pred[i])
