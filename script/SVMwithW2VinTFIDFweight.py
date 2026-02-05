import gensim
import numpy as np
import pandas as pd
import MeCab

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# ===== モデル & MeCab =====
model = gensim.models.Word2Vec.load("word2vec_review.model")

tagger = MeCab.Tagger(
    "-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/ipadic -Owakati"
)

def wakati(text):
    return tagger.parse(text).strip()


# ===== TF-IDF + Word2Vec 文ベクトル =====
def sentence_vector_tfidf(sentence, model, tfidf_vec, feature_names):
    words = sentence.split()

    vec = np.zeros(model.vector_size)
    weight_sum = 0.0

    for w in words:
        if w in model.wv and w in feature_names:
            idx = feature_names[w]
            weight = tfidf_vec[idx]
            vec += weight * model.wv[w]
            weight_sum += weight

    if weight_sum == 0:
        return vec
    return vec / weight_sum


def build_dataset(df, model, vectorizer):
    wakatis = df["text"].apply(wakati).tolist()
    tfidf = vectorizer.transform(wakatis)

    feature_names = {
        word: idx for idx, word in enumerate(vectorizer.get_feature_names_out())
    }

    X = np.vstack([
        sentence_vector_tfidf(
            wakatis[i],
            model,
            tfidf[i].toarray().flatten(),
            feature_names
        )
        for i in range(len(wakatis))
    ])
    return X


# ===== データ読み込み =====
df = pd.read_csv(
    "data/pn.tsv",
    sep="\t",
    header=None,
    names=["num", "label", "text", "judges", "usage"]
)

df_train = df[df["usage"] == "train"]
df_dev   = df[df["usage"] == "dev"]
df_test  = df[df["usage"] == "test"]

train_y = df_train["label"].values
dev_y   = df_dev["label"].values
test_y  = df_test["label"].values


# ===== TF-IDF 学習（重み用） =====
vectorizer = TfidfVectorizer(
    min_df=2,
    max_df=0.9
)
vectorizer.fit(df_train["text"].apply(wakati))


# ===== ベクトル化 =====
train_X = build_dataset(df_train, model, vectorizer)
dev_X   = build_dataset(df_dev, model, vectorizer)
test_X  = build_dataset(df_test, model, vectorizer)


# ===== 分類器 =====
clf = Pipeline([
    ("svm", LinearSVC(
        C=0.05,
        max_iter=10000,
        class_weight="balanced",
        multi_class="ovr"
    ))
])

clf.fit(train_X, train_y)


# ===== 評価 =====
print("train:", accuracy_score(train_y, clf.predict(train_X)))
print("dev  :", accuracy_score(dev_y, clf.predict(dev_X)))
print("test :", accuracy_score(test_y, clf.predict(test_X)))


# ===== 誤分類保存 =====
y_pred = clf.predict(dev_X)
wrong_idx = np.where(y_pred != dev_y)[0]

print("誤分類数:", len(wrong_idx))

np.save("word2vec_tfidf_widx.npy", wrong_idx)
np.save("word2vec_tfidf_dev_pred.npy", y_pred)

def print_wrong_examples(df_dev, wrong_idx):
    for idx in wrong_idx:
        true_label = dev_y[idx]
        predicted_label = y_pred[idx]
        text = df_dev.iloc[idx]['text']
        print(f"Index: {idx}, True Label: {true_label}, Predicted Label: {predicted_label}")
        print(f"Text: {text}\n")

#print_wrong_examples(df_dev, wrong_idx)


def analyze(idxs):
    ll = [0, 0, 0]
    true_label = dev_y[idxs]
    for l in true_label:
        ll[l + 1] += 1
    print(f"ネガ:{ll[0]}, ニュートラル:{ll[1]}, ポジ:{ll[2]}")
#analyze(list(wrong_idx))
