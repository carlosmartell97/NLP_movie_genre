#!/usr/local/bin/python3
import pandas as pd
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
import warnings
import data_and_scores


df_all_genres = pd.read_csv('tmdb_5000_movies.csv')  # https://www.kaggle.com/tmdb/tmdb-movie-metadata#tmdb_5000_movies.csv
df = pd.read_csv('moviesResults3.csv')  # solo un género por película, agrupando géneros parecidos
df_all_genres, df = data_and_scores.clean_data(df_all_genres, df)

genre_id_df = df[['genre', 'genre_id']].drop_duplicates().sort_values('genre_id')
genre_to_id = dict(genre_id_df.values)
print(df.head(10), end="\n\n")  # show first 10 rows
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
    (RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0), ("random forest", [])),
    (LinearSVC(), ("linear support vector", [])),
    (MultinomialNB(), ("multinomial naive bayes", [])),
    (LogisticRegression(random_state=0), ("logistic regression", []))
]
print("trying different test sizes...")
test_sizes = np.arange(0.20, 0.41, 0.005)
best_score = 0
best_model = "?"
best_test_size = 0
best_classifier = None
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
for size in test_sizes:
    print("test: %d%%" % (size*100))
    X_train, X_test, y_train, y_test = train_test_split(df['synopsis'], df['genre'], random_state=0, test_size=size)
    X_train_counts = count_vect.fit_transform(X_train)
    X_test_counts = count_vect.transform(X_test)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    for m in models:
        model = m[0]
        model_name = m[1][0]
        model_scores = m[1][1]
        clf = model.fit(X_train_tfidf, y_train)
        score = model.score(X_test_tfidf, y_test)
        # score = data_and_scores.calc_score(clf, X_test, y_test, df_all_genres, df, count_vect)
        model_scores.append(score)
        if score > best_score:
            best_score = score
            best_model = model_name
            best_test_size = size
            best_classifier = clf
        print("\t%s score:%f" % (model_name, score))
print("best model:%s with %0.2f%% accuracy, using %d%% for testing" % (best_model, best_score*100, best_test_size*100))
for m in models:
    model_name = m[1][0]
    model_scores = m[1][1]
    plt.plot(test_sizes, model_scores, label=model_name)
plt.xlabel('test set %'), plt.ylabel('accuracy'), plt.title('model accuracies'), plt.grid(True), plt.legend(), plt.show()

while True:
    input_synopsis = input("Enter movie description: ")
    prediction = best_classifier.predict(count_vect.transform([input_synopsis]))
    print("predicted genre: %s" % prediction)
