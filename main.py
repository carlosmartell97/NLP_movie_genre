#!/usr/local/bin/python3
import pandas as pd
import json

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
df = df[df.synopsis != '']  # drop all movies with no synopsis
for _index, row in df.iterrows():  # only consider top genre from all genres in movie
    row['genre'] = json.loads(row['genre'])[0]['name']
print(df.head()), print("")
