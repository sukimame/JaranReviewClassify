import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

tfidf = np.load('data/tfidf.npz', allow_pickle=True)

train_X, train_y = tfidf['X_train'], tfidf['y_train']
dev_X, dev_y = tfidf['X_dev'], tfidf['y_dev']
test_X, test_y = tfidf['X_test'], tfidf['y_test']

model = SVC(kernel='linear', random_state=None)
model.fit(train_X, train_y)

# トレーニングデータに対する精度
pred_train = model.predict(train_X)
accuracy_train = accuracy_score(train_y, pred_train)
print('トレーニングデータに対する正解率： %.2f' % accuracy_train)
