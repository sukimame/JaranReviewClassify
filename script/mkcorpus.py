import MeCab
import pandas as pd
tagger = MeCab.Tagger(
    "-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/ipadic -Owakati"
)

df = pd.read_csv('data/pn.tsv', sep="\t", header=None, names=['num', 'label', 'text', 'judges', 'usage'])
df_train = df[df['usage'] == 'train']

with open("data/corpus.txt", "w", encoding="utf-8") as fout:
    for line in df_train['text']:
        line = line.strip()
        if not line:
            continue
        wakati = tagger.parse(line).strip()

        fout.write(wakati + "\n")


