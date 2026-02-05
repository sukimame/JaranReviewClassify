import numpy as np
import pandas as pd


df = pd.read_csv('data/pn.tsv', sep="\t", header=None, names=['num', 'label', 'text', 'judges', 'usage'])
df_dev   = df[df['usage'] == 'dev']
dev_y = df_dev['label'].values

def print_wrong_examples(df_dev, wrong_idx, y_pred):
    for idx in wrong_idx:
        true_label = dev_y[idx]
        predicted_label = y_pred[idx]
        text = df_dev.iloc[idx]['text']
        print(
            f"{idx:>4} | "
            f"T:{true_label:^3} "
            f"P:{predicted_label:^3} | "
            f"{text}"
        )


tfidf = np.load('tfidf_widx.npy')
w2v = np.load('word2vec_widx.npy')

tfidf_pred = np.load('tfidf_dev_pred.npy')
w2v_pred = np.load('word2vec_dev_pred.npy')

tt = set(tfidf) & set(w2v)
tf = set(tfidf) - tt
ft = set(w2v) - tt

N = 10
print("両方誤分類:", len(tt))
for idx in list(tt)[:N]:
    true_label = dev_y[idx]
    pred1 = tfidf_pred[idx]
    pred2 = w2v_pred[idx]
    text = df_dev.iloc[idx]['text']
    print(
        f"{idx:>4} | "
        f"T:{true_label:^3} "
        f"TF:{pred1:^3} "
        f"W2V:{pred2:^3} | "
        f"{text}"
    )


print("TF-IDFのみ誤分類:", len(tf))
print_wrong_examples(df_dev, list(tf)[:N], tfidf_pred)
print("Word2Vecのみ誤分類:", len(ft))
print_wrong_examples(df_dev, list(ft)[:N], w2v_pred)

def analyze(idxs):
    ll = [0, 0, 0]
    true_label = dev_y[idxs]
    for l in true_label:
        ll[l + 1] += 1
    print(f"ネガ:{ll[0]}, ニュートラル:{ll[1]}, ポジ:{ll[2]}")

analyze(list(tt))
analyze(list(tf))
analyze(list(ft))

analyze([i for i in range(dev_y.shape[0])])  # 全体