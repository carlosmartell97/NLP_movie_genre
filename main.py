#!/usr/local/bin/python3
import pandas as pd
import json
import numpy as np

import matplotlib.pyplot as plt
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


df = pd.read_csv('tmdb_5000_movies.csv')  # https://www.kaggle.com/tmdb/tmdb-movie-metadata#tmdb_5000_movies.csv

#  clean data
ignore_columns = ['budget', 'homepage', 'id', 'keywords', 'original_language',
                'popularity', 'production_companies', 'production_countries',
                'release_date', 'revenue', 'runtime', 'spoken_languages',
                'status', 'tagline', 'vote_average', 'vote_count',
                'original_title']
for column in ignore_columns:  # drop all extra columns
    df = df.drop(column, axis='columns')
df = df.rename({'overview':'synopsis', 'genres':'genre'}, axis='columns')  # rename overview and genres columns
df = df[df.genre != '[]']  # drop all movies with no genres
df = df[pd.notnull(df['synopsis'])]  # drop all movies with no synopsis
for _index, row in df.iterrows():  # only consider top genre from all genres in movie
    row['genre'] = json.loads(row['genre'])[0]['name']
df['genre_id'] = df['genre'].factorize()[0]  # new column with genre as number
genre_id_df = df[['genre', 'genre_id']].drop_duplicates().sort_values('genre_id')
genre_to_id = dict(genre_id_df.values)
print(genre_to_id)
print(df.head(10)), print("")  # show first 10 rows

# fig = plt.figure(figsize=(8, 6))
df.groupby('genre').synopsis.count().plot.bar(ylim=0)
plt.show()


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3, norm='l2',
                        encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')
features = tfidf.fit_transform(df.synopsis).toarray()
labels = df.genre_id
print(features.shape, end='\n\n')


N = 2
for genre, genre_id in sorted(genre_to_id.items()):
    features_chi2 = chi2(features, labels == genre_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(genre))
    print("  . Most correlated unigrams:\n    . {}".format('\n    . '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n    . {}".format('\n    . '.join(bigrams[-N:])))


# Naive Bayes Classifier
X_train, X_test, y_train, y_test = train_test_split(df['synopsis'], df['genre'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
test_synopsis = "No one could have expected what was behind that door. No one could have warned them about the nightmare they were all about to face."
test_prediction = clf.predict(count_vect.transform([test_synopsis]))
print("\n%s -> %s\n" % (test_synopsis, test_prediction))

# Logistic Regression, Multinomal Naive Bayes,
# Linear Support Vector Machine and Random Forest
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

print(cv_df.groupby('model_name').accuracy.mean())
