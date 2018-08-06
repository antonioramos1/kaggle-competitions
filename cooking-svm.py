# intial fork from https://github.com/SaquibAhmad/Kaggle/blob/master/whats-cooking/source/svm_model.ipynb
import pandas as pd
import json
import numpy as np
import re
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from nltk.stem.porter import PorterStemmer

train = json.load(open('C:/Users/heret/Downloads/whats-cooking/train.json'))
test = json.load(open('C:/Users/heret/Downloads/whats-cooking/test.json'))
test_id = [recipe['id'] for recipe in test]

train_text = [' '.join(recipe['ingredients']) for recipe in train]
test_text = [' '.join(recipe['ingredients']) for recipe in test]
target = [recipe['cuisine'] for recipe in train]

train_text = [" ".join([re.sub(r"\d+", "", word) for word in recipe.split(" ")])
              for recipe in train_text] #removing numbers from ingredient list
test_test = [" ".join([re.sub(r"\d+", "", word) for word in recipe.split(" ")])
             for recipe in test_text]

train_text = [' '.join([PorterStemmer().stem(word) for word in recipe.split(' ')]) for recipe in train_text]
test_text = [' '.join([PorterStemmer().stem(word) for word in recipe.split(' ')]) for recipe in test_text]

tfidf = TfidfVectorizer(binary=True, lowercase=True, stop_words='english', ngram_range=(1,1))
X = tfidf.fit_transform(train_text).astype('float16')
X_test = tfidf.transform(test_text).astype('float16')
label_enc = LabelEncoder()
y = label_enc.fit_transform(target)

# number_ingredients_train = [len([word for word in recipe.split(' ')])
#                             for recipe in train_text] #using number of ingredientes per recipy as a new variable
# number_ingredients_test = [len([word for word in recipe.split(' ')])
#                            for recipe in test_text]
# X = np.hstack((X.todense(), np.array(number_ingredients_train).reshape(39774,1))).astype("float16") #39774
# X_test = np.hstack((X_test.todense(), np.array(number_ingredients_test).reshape(9944,1))).astype("float16")
# X = sparse.csr_matrix(X, dtype=np.float16)
# X_test = sparse.csr_matrix(X_test, dtype=np.float16)

model = OneVsRestClassifier(
    SVC(C=100, kernel='rbf', gamma=1, decision_function_shape=None, random_state=2018),
    n_jobs=4
)

model.fit(X, y)

y_test = model.predict(X_test) #0.81 acc LB
y_pred = label_enc.inverse_transform(y_test)

sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('svm_output_v2.csv', index=False)
