
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings
from itertools import product
import pickle


def queryInput(user_input):
 

    result_params_list_df_ps = pd.DataFrame(columns=['movieId', 'title', 'genres', 'BM25_score'])

    res = es.search(index="movies", body={"query": {"match": {"title":"{}".format(user_input)}}}, size = 1000)
    hits = res['hits']['total']['value']
    print("Got {} Hits:".format(hits))

    
    try :
        for i in range(hits):
            temp_df = pd.DataFrame([ [ int(res['hits']['hits'][i]['_source']['movieId']), res['hits']['hits'][i]['_source']['title'], res['hits']['hits'][i]['_source']['genres'], res['hits']['hits'][i]['_score'] ] ], columns=['movieId', 'title', 'genres', 'BM25_score'])
            result_params_list_df_ps = result_params_list_df_ps.append(temp_df, ignore_index = True)
    except:
        for i in range(1000):
            temp_df = pd.DataFrame([ [ int(res['hits']['hits'][i]['_source']['movieId']), res['hits']['hits'][i]['_source']['title'], res['hits']['hits'][i]['_source']['genres'], res['hits']['hits'][i]['_score'] ] ], columns=['movieId', 'title', 'genres', 'BM25_score'])
            result_params_list_df_ps = result_params_list_df_ps.append(temp_df, ignore_index = True)
    return  result_params_list_df_ps


def preProcessing():
  
    print('\nPlease wait while we do some pre-processing.....')
    rating_df = pd.read_csv('./data/ratings.csv')  
   
    movie_rating_df_pu = rating_df[['userId','movieId','rating']]

    movie_avg_rating_df = movie_rating_df_pu.groupby(by='movieId').mean()
    movie_avg_rating_df = movie_avg_rating_df.drop('userId', axis=1).reset_index()
    
  
    clustered_users_df = clusterUsers(movie_rating_df_pu,movie_avg_rating_df)
    filled_user_ratings_df = fillUserRatings(clustered_users_df) 


    print("\nSaving the clustering result...")
    pickle.dump(filled_user_ratings_df, open("./data/user_ratings_after_clustering.p", "wb"))

    return movie_avg_rating_df, filled_user_ratings_df


def startLoop(movie_avg_rating_df, movie_rating_df_pu):
 

    print('\ntype "//exit" if you want to exit the search')
    user_input_movie = input("Which movie do you want? (by title): ")
    
    while( user_input_movie != '//exit' ) :
        
        user_input_user = input("\nFor which user do you want to search? (int): ")
        while ( ( user_input_user.isdigit() == False) ):
            user_input_user = input("\nFor which user do you want to search? (int): ")

      
        query_result_params_df = queryInput(user_input_movie)
       
        final_df = finalRanking(query_result_params_df, movie_avg_rating_df, movie_rating_df_pu, user_input_user)
     
        print(final_df)


        print('type "//exit" if you want to exit the search')
        user_input_movie = input("Which movie do you want? (by title): \n")
    return


def finalRanking(query_result_params_df, movie_avg_rating_df, movie_rating_df_pu, user_input_user):
 
    final_df = query_result_params_df.copy(deep=True)

    final_df = final_df.merge(movie_avg_rating_df, on = 'movieId', how = 'left')
    final_df.rename(columns = {'rating':'avg_rating'}, inplace=True)
    
  
    temp = movie_rating_df_pu.copy(deep=True)
   
    temp.drop(temp[temp['userId'] != int(user_input_user)].index, inplace=True)
  
    final_df = final_df.merge(temp, on = 'movieId', how = 'left')
    final_df.rename(columns = {'rating':'user_rating_after_clustering'}, inplace=True)

  
    final_df['final_score'] = np.nan
    #first pass 
    final_df['final_score'] = final_df['BM25_score'] + final_df['avg_rating'] + final_df['user_rating_after_clustering']
   
    final_df['final_score'].fillna(final_df['BM25_score'] + final_df['avg_rating'], inplace=True)
 
    final_df['final_score'].fillna(final_df['BM25_score'], inplace=True)
    
  
    final_df = final_df.sort_values(by = 'final_score', ascending=False).reset_index()
    final_df.drop(['index','genres','userId'], axis=1, inplace=True)
    final_df.drop_duplicates(inplace=True)

    return final_df


def fillUserRatings(clustered_users_df):
    
    filled_user_ratings_df = clustered_users_df.copy(deep=True)

    for cluster in filled_user_ratings_df['cluster'].unique():
       
        temp = filled_user_ratings_df[filled_user_ratings_df['cluster'] == cluster]
        temp.fillna(temp.mean(), inplace=True)
      
        filled_user_ratings_df[filled_user_ratings_df['cluster'] == cluster] = temp

   
    filled_user_ratings_df = filled_user_ratings_df.reset_index().drop('cluster', axis=1).melt('userId', var_name='movieId', value_name='rating').sort_values(by=['userId','movieId'])
    return filled_user_ratings_df


def combineWithCluster(df, cluster_labels):
   
    df['cluster'] = pd.Series(cluster_labels, index=df.index)
    return 


def cartesianProduct(movie_rating_df_pu):
   
    l1 = list(movie_rating_df_pu['userId'].unique())
    l2 = list(movie_rating_df_pu['movieId'].unique())
    temp = pd.DataFrame(list(product(l1, l2)), columns=['userId', 'movieId'])
    temp.sort_values(by=['userId','movieId']).reset_index(inplace=True, drop=True)
    temp = temp.merge(movie_rating_df_pu, on = ['userId','movieId'], how='left')
    return temp


def fillNanWithAvgGenreRating(movie_rating_df_pu, movie_avg_rating_df):
  
    user_movie_product_df = cartesianProduct(movie_rating_df_pu)

 
    movie_details_df = pd.read_csv('./data/movies.csv')
    avg_rating_per_genre = movie_avg_rating_df.merge(movie_details_df[['movieId','genres']], on='movieId', how='left')
    avg_rating_per_genre.drop('movieId',axis=1,inplace=True)
    avg_rating_per_genre = avg_rating_per_genre.groupby(by='genres').mean()
    avg_rating_per_genre.rename(columns={'rating':'avg_rating_per_genre'},inplace=True)
    
    movie_rating_df_pu_with_genre = user_movie_product_df.merge(movie_details_df[['movieId','genres']], on='movieId', how='left')
    movie_rating_df_pu_with_genre = movie_rating_df_pu_with_genre.merge(avg_rating_per_genre, on='genres',how='left')
    movie_rating_df_pu_with_genre['rating'] = movie_rating_df_pu_with_genre.rating.fillna(movie_rating_df_pu_with_genre.avg_rating_per_genre)
    movie_rating_df_pu_noNaN = movie_rating_df_pu_with_genre.drop(['avg_rating_per_genre','genres'], axis=1)

    return movie_rating_df_pu_noNaN

def getMostRatedMovieColumns(user_movie_ratings, max_number_of_movies):
   
    temp = user_movie_ratings.copy(deep=True)
    temp = temp.append(user_movie_ratings.count(), ignore_index=True)
    temp.index = range(1,len(temp)+1)
    temp_sorted = temp.sort_values(len(temp), axis=1, ascending=False)
    temp_sorted = temp_sorted.drop(temp_sorted.tail(1).index)
    return_df = temp_sorted.iloc[:, :max_number_of_movies]
    return return_df.columns


def
    movie_rating_df_pu_noNaN = fillNanWithAvgGenreRating(movie_rating_df_pu, movie_avg_rating_df)
    X = movie_rating_df_pu.pivot(index='userId', columns='movieId', values='rating') 
    X_noNaN = movie_rating_df_pu_noNaN.pivot(index='userId', columns='movieId', values='rating')
    best_movie_columns = getMostRatedMovieColumns(X,1000)
    X_best_noNaN = X_noNaN[best_movie_columns]
    
    predictions = predictWithKmeans(6, X_best_noNaN)   
    combineWithCluster(X, predictions)
    return X

def predictWithKmeans(clusters, matrix):
    
    return KMeans(n_clusters = clusters, algorithm='full', random_state=2).fit_predict(matrix)



if __name__ == "__main__":
   
    warnings.simplefilter("ignore")

   
    es = Elasticsearch()
    movie_avg_rating_df, movie_rating_df_pu = preProcessing()
    startLoop(movie_avg_rating_df, movie_rating_df_pu)