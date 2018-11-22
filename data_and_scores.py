#!/usr/local/bin/python3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import json


def clean_data(df_all_genres, df):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    df = df.rename({'overview': 'synopsis'}, axis='columns')  # rename overview and genres columns
    df_all_genres = df_all_genres.rename({'overview': 'synopsis'}, axis='columns')  # rename overview and genres columns
    df = df[df.genre != '[]']  # drop all movies with no genres
    df_all_genres = df_all_genres[df_all_genres.genres != '[]']  # drop all movies with no genres
    df = df[pd.notnull(df['synopsis'])]  # drop all movies with no synopsis
    df_all_genres = df_all_genres[pd.notnull(df_all_genres['synopsis'])]  # drop all movies with no synopsis
    for _index, row in df.iterrows():  # stem words and remove stop words
        words = word_tokenize(row['synopsis'])
        stemmed_without_stops = []
        for w in words:
            if not w in stop_words:
                stemmed_without_stops.append(ps.stem(w))
        row['synopsis'] = ' '.join(word for word in stemmed_without_stops)
    df['genre_id'] = df['genre'].factorize()[0]  # new column with genre as number
    return (df_all_genres, df)


def calc_score(model, X_test, y_test, df_all_genres, df, count_vect):
    # print("\tX_test"), print(X_test, end="\n\n")
    # print("\ty_test"), print(y_test, end="\n\n")
    total_tests = len(y_test.axes[0])
    tests_correct = 0
    for i, axis in enumerate(y_test.axes[0]):
        pos = df_all_genres['title'] == df.loc[axis]['title']
        expected_genres = json.loads(df_all_genres.loc[pos]['genres'].values[0])
        test_synopsis = X_test[axis]
        predicted = model.predict(
            count_vect.transform([test_synopsis])
        )[0].split('/')
        # print("synopsis: %s" % test_synopsis)
        # print("predicted: %s" % predicted)
        # print("expected: %s" % expected_genres)
        found = False
        for predicted_genre in predicted:
            for expected_genre in expected_genres:
                if expected_genre['name'] == predicted_genre:
                    found = True
        if found:
            tests_correct += 1
    score = tests_correct / total_tests
    return score
