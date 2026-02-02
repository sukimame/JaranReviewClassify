# JaranReviewClassify
classify Japanese review from Jaran

本リポジトリには，Creative Commons BY-NC-SA 4.0 で公開されている
じゃらん口コミデータセットは含まれていない。


目的
じゃらんのレビューについて、SVMを用いた分類を行う。

データセット
用いるデータセットはJapanese Realistic Textual Entailment Corpusのpn.tsvにある感情極性ラベルが付加されたレビューである。

data/pn.tsv

Data for sentiment analysis.

#	Explanation	Samples
0	ID of the example	pnXYZq00001
1	Label	1 (Positive), 0 (Neutral), -1 (Negative)
2	Text	駅まで近い。
3	Judges (JSON format)	{"0": 1, "1": 4}
4	Usage	train, dev, test

データの構造は上記の通り。

訓練データ
3888
開発用検証データ
1112
検証用データ
553

https://gotutiyan.hatenablog.com/entry/2020/09/10/181919#TfidfVectorizerの入出力






