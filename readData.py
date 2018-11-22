import csv
import json

finalMovies = {}

with open('moviesDb.csv', newline='') as csvfile:
    movieReader = csv.DictReader(csvfile)
    counter = 0
    for row in movieReader:
        #finalMovies[row['title']] = {}
        movieTitle = row['title']
        movieGenres = json.loads(row['genres'])
        movieOverview = row['overview']
        movieKeyWords = json.loads(row['keywords'])

        arrayKeywords = []
        for j in range(0,len(movieKeyWords)):
            arrayKeywords.append(movieKeyWords[j]['name'])

        for i in range(0,len(movieGenres)):
            if(movieGenres[0]['name'] == "Horror" or movieGenres[0]['name'] == "Thriller" or movieGenres[0]['name'] == "Crime"):
                finalMovies[movieTitle] = movieTitle, "Horror/Thriller/Crime", movieOverview, arrayKeywords
            elif(movieGenres[0]['name'] == "Fantasy" or movieGenres[0]['name'] == "Family" or movieGenres[0]['name'] == "Animation"):
                finalMovies[movieTitle] = movieTitle, "Fantasy/Family/Animation", movieOverview, arrayKeywords
            elif(movieGenres[0]['name'] == "Drama" or movieGenres[0]['name'] == "Action" or movieGenres[0]['name'] == "Adventure"):
                finalMovies[movieTitle] = movieTitle, movieGenres[0]['name'], movieOverview, arrayKeywords
            else:
                finalMovies[movieTitle] = movieTitle, "Other", movieOverview, arrayKeywords
            #finalMovies[movieTitle] = movieTitle, movieGenres[0]['name'], movieOverview, arrayKeywords
            #counter += 1

        arrayKeywords = []

with open('moviesResults2.csv','w',newline='') as csvwrite:
    fieldnames=['title','genre','overview','keywords']
    writer = csv.DictWriter(csvwrite,fieldnames=fieldnames)

    writer.writeheader()
    for key in finalMovies:
        #writer.writerow({'title':finalMovies[k][0],'genre':finalMovies[k][1],'overview':finalMovies[k][2],'keywords':finalMovies[k][3]})
        writer.writerow({'title':finalMovies[key][0],'genre':finalMovies[key][1],'overview':finalMovies[key][2],'keywords':finalMovies[key][3]})
