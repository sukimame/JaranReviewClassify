import gensim

with open("data/corpus.txt", "r", encoding="utf-8") as fin:
    sentences = [line.strip().split() for line in fin if line.strip()]
print(sentences[:2])

model = gensim.models.Word2Vec.load('ja/ja.bin')

model.build_vocab(sentences, update=True)
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

model.save("word2vec_review.model")