from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np


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
    
    rating_df = pd.read_csv('./data/ratings.csv')  
    #pu = per user
    movie_rating_df_pu = rating_df[['userId','movieId','rating']]
    movie_avg_rating_df = movie_rating_df_pu.groupby(by='movieId').mean()
    movie_avg_rating_df = movie_avg_rating_df.drop('userId', axis=1).reset_index()
    return movie_avg_rating_df, movie_rating_df_pu


def startLoop(movie_avg_rating_df, movie_rating_df_pu):


    print('type "//exit" if you want to exit the search')
    user_input_movie = input("Which movie do you want? (by title): \n")
    
    while( user_input_movie != '//exit' ) :
        
        user_input_user = input("For which user do you want to search? (int): \n")
        while ( ( user_input_user.isdigit() == False) ):
            user_input_user = input("For which user do you want to search? (int): \n")

      
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
    final_df.rename(columns = {'rating':'user_rating'}, inplace=True)

    final_df['final_score'] = np.nan
  
    final_df['final_score'] = final_df['BM25_score'] + final_df['avg_rating'] + final_df['user_rating']

    final_df['final_score'].fillna(final_df['BM25_score'] + final_df['avg_rating'], inplace=True)
 
    final_df['final_score'].fillna(final_df['BM25_score'], inplace=True)
    
 
    final_df = final_df.sort_values(by = 'final_score', ascending=False).reset_index()
    final_df.drop(['index','genres','userId'], axis=1, inplace=True)
    final_df.drop_duplicates(inplace=True)


    return final_df

if __name__ == "__main__":

    es = Elasticsearch()
  
    movie_avg_rating_df, movie_rating_df_pu = preProcessing()
 
    startLoop(movie_avg_rating_df, movie_rating_df_pu)