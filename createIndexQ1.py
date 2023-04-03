from elasticsearch import helpers, Elasticsearch
import csv

es = Elasticsearch()

with open('./data/movies.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    helpers.bulk(es, reader, index='movies', doc_type='my-type')

