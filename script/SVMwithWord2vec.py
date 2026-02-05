import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import MeCab

model = gensim.models.Word2Vec.load('word2vec_review.model')

tagger = MeCab.Tagger(
    "-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/ipadic -Owakati"
)

#print(model.wv.most_similar("残念")) #モデルに語彙を加えて再学習したため、「残念」でもベクトルが取れる。

def sentence_vector(sentence, model):
    words = sentence.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# mecabで分かち書きし、文ベクトルのデータセットを作成する関数
def build_dataset(df, model):
    dataset = np.vstack(
        [
            sentence_vector(tagger.parse(line).strip(), model) for line in df['text']
        ]
    )
    return dataset

df = pd.read_csv('data/pn.tsv', sep="\t", header=None, names=['num', 'label', 'text', 'judges', 'usage'])
df_train = df[df['usage'] == 'train']
df_dev   = df[df['usage'] == 'dev']
df_test  = df[df['usage'] == 'test']

train_X = build_dataset(df_train, model)
dev_X   = build_dataset(df_dev, model)
test_X  = build_dataset(df_test, model)

train_y = df_train['label'].values
dev_y   = df_dev['label'].values
test_y  = df_test['label'].values


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

np.save('word2vec_widx.npy', wrong_idx)
np.save('word2vec_dev_pred.npy', y_pred)

def print_wrong_examples(df_dev, wrong_idx):
    for idx in wrong_idx:
        true_label = dev_y[idx]
        predicted_label = y_pred[idx]
        text = df_dev.iloc[idx]['text']
        print(f"Index: {idx}, True Label: {true_label}, Predicted Label: {predicted_label}")
        print(f"Text: {text}\n")

# print_wrong_examples(df_dev, wrong_idx)
