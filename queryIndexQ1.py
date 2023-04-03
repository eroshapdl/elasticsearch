
from elasticsearch import Elasticsearch

def queryInput(user_input):
    res = es.search(index="movies", body={"query": {"match": {"title":"{}".format(user_input)}}}, size = 1000)
    hits = res['hits']['total']['value']
    print("Got {} Hits:".format(hits))

  
    try :
        for i in range(hits):
            print(i+1,') ',res['hits']['hits'][i]['_source']['title'])
    except:
        for i in range(1000):
            print(i+1,') ',res['hits']['hits'][i]['_source']['title'])
            
    return
    


es = Elasticsearch()

print('type "//exit" if you want to exit the search')
user_input = input("Which movie do you want? (by title): \n")

while(user_input != '//exit'):
    queryInput(user_input)
    print('type "//exit" if you want to exit the search')
    user_input = input("Which movie do you want? (by title): \n")

    

