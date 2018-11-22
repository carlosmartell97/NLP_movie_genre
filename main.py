#!/usr/local/bin/python3
import pandas as pd
import json
import numpy as np
from operator import attrgetter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import seaborn as sns
import warnings


def get_key(item):
    return item[1]


df = pd.read_csv('moviesResults3.csv')  # solo un género por película, agrupando géneros parecidos

#  clean data
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
df = df.rename({'overview': 'synopsis'}, axis='columns')  # rename overview and genres columns
df = df[df.genre != '[]']  # drop all movies with no genres
df = df[pd.notnull(df['synopsis'])]  # drop all movies with no synopsis
for _index, row in df.iterrows():  # stem words and remove stop words
    words = word_tokenize(row['synopsis'])
    stemmed_without_stops = []
    for w in words:
        if not w in stop_words:
            stemmed_without_stops.append(ps.stem(w))
    row['synopsis'] = ' '.join(word for word in stemmed_without_stops)
df['genre_id'] = df['genre'].factorize()[0]  # new column with genre as number
genre_id_df = df[['genre', 'genre_id']].drop_duplicates().sort_values('genre_id')
genre_to_id = dict(genre_id_df.values)
print(df.head(10)), print("")  # show first 10 rows
df.groupby('genre').synopsis.count().plot.bar(ylim=0)
plt.title('word count per genre'), plt.show()

# extract features
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3, norm='l2',
                        encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')
features = tfidf.fit_transform(df.synopsis).toarray()
labels = df.genre_id
print(features.shape, end='\n\n')

# show most correlated unigrams and bigrams
# N = 2
# for genre, genre_id in sorted(genre_to_id.items()):
#     features_chi2 = chi2(features, labels == genre_id)
#     indices = np.argsort(features_chi2[0])
#     feature_names = np.array(tfidf.get_feature_names())[indices]
#     unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#     bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#     print("# '{}':".format(genre))
#     print("  . Most correlated unigrams:\n    . {}".format('\n    . '.join(unigrams[-N:])))
#     print("  . Most correlated bigrams:\n    . {}".format('\n    . '.join(bigrams[-N:])))

# various models
warnings.filterwarnings("ignore")
models = [
    (RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0), "random forest"),
    (LinearSVC(), "linear support vector"),
    (MultinomialNB(), "multinomial naive bayes"),
    (LogisticRegression(random_state=0), "logistic regression")
]
print("trying different test sizes...")
test_sizes = np.arange(0.25, 0.4, 0.01)
best_score = 0
best_model = "?"
best_test_size = 0
for size in test_sizes:
    print("test: %d%%" % (size*100))
    for m in models:
        model = m[0]
        model_name = m[1]
        X_train, X_test, y_train, y_test = train_test_split(df['synopsis'], df['genre'], random_state=0, test_size=size)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        X_test_counts = count_vect.transform(X_test)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        X_test_tfidf = tfidf_transformer.transform(X_test_counts)
        clf = model.fit(X_train_tfidf, y_train)
        score = model.score(X_test_tfidf, y_test)
        if score > best_score:
            best_score = score
            best_model = model_name
            best_test_size = size
        print("\t%s score:%f" % (model_name, score))
        # test_synopsis = "Summer's here and everyone is pumped for the beach. They weren't accounting for a speaking dog to join them though, and change their lives forever."
        # test_prediction = clf.predict(count_vect.transform([test_synopsis]))
        # print("\n%s -> %s\n" % (test_synopsis, test_prediction))
print("best model:%s with %0.2f%% accuracy, using %d%% for testing" % (best_model, best_score*100, best_test_size*100))
