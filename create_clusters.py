import pandas as pd
import csv
import tqdm
import json
import re
import ast
import os
import torch

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np


os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set this to the number of cores you want to use
os.environ['OMP_NUM_THREADS'] = '1'

def create_clusters(embeddings_dataset, model):
    # all_embeddings = numpy array of all embeddings
    all_embeddings = []
    embeddings_dataset['cluster_filter'] = None
    i = 0
    same = 0
    total = 0
    for index, row in tqdm.tqdm(embeddings_dataset.iterrows(), total=embeddings_dataset.shape[0]):
        embeddings = []
        title_embedding = model.encode(row['SubmissionTitle'])
        abstract_embedding = model.encode(row['SubmissionAbstract'])
        s_id = row['SubmissionID']
    
        
        embeddings.append(np.concatenate((title_embedding, abstract_embedding), axis=None))
        
        
        # # Ensure embeddings are not zero-dimensional
        # if isinstance(title_embedding, str):
        #     title_embedding = ast.literal_eval(title_embedding)
        # if isinstance(abstract_embedding, str):
        #     abstract_embedding = ast.literal_eval(abstract_embedding)
        # if isinstance(doi_embedding, str):
        #     doi_embedding = ast.literal_eval(doi_embedding)
        
        # if not title_embedding or not abstract_embedding or not doi_embedding:
        #     print(f"Zero-dimensional embedding found at index {index}")
        #     continue
        
        row['authorPublicationHistory_embedding'] = ast.literal_eval(row['authorPublicationHistory_embedding'])
        embeddings_dataset.at[index, 'authorPublicationHistory_embedding'] = row['authorPublicationHistory_embedding']
        
        row['authorPublicationHistory'] = ast.literal_eval(row['authorPublicationHistory'])
        embeddings_dataset.at[index, 'authorPublicationHistory'] = row['authorPublicationHistory']
        for authorWorks in row['authorPublicationHistory']:
            authorTitleEmbedding = model.encode(authorWorks['title'])
            authorAbstractEmbedding = model.encode(authorWorks['abstract'])
            
            embeddings.append(np.concatenate((authorTitleEmbedding, authorAbstractEmbedding), axis=None))            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            # combined_embedding_author = []
            # author_title_embedding = authorWorks['title_embedding']
            # author_abstract_embedding = authorWorks['abstract_embedding']
            # author_doi_embedding = authorWorks['doi_embedding']
            
            # print(author_title_embedding.shape)
            # exit()
            # # Convert to numpy arrays
            # author_title_embedding = np.array(author_title_embedding)
            # author_abstract_embedding = np.array(author_abstract_embedding)
            # author_doi_embedding = np.array(author_doi_embedding)
            
            # # # Ensure embeddings are not zero-dimensional
            # # if isinstance(author_title_embedding, str):
            # #     author_title_embedding = ast.literal_eval(clean_array_string(author_title_embedding))
            # # if isinstance(author_abstract_embedding, str):
            # #     author_abstract_embedding = ast.literal_eval(clean_array_string(author_abstract_embedding))
            # # if isinstance(author_doi_embedding, str):
            # #     author_doi_embedding = ast.literal_eval(clean_array_string(author_doi_embedding))
            
            # # if not author_title_embedding or not author_abstract_embedding or not author_doi_embedding:
            # #     print(f"Zero-dimensional author embedding found at index {index}")
            # #     continue
            
            
            # combined_embedding_author = np.concatenate((author_title_embedding, author_abstract_embedding, author_doi_embedding), axis=None)
            # embeddings.append(combined_embedding_author)
        
        if len(embeddings) == 1:
            k = 0
        if len(embeddings) == 2:
            k = 1
        if len(embeddings) >= 3:
            k = 2
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        submission_cluster = clusters[0]
        clusters = clusters[1:]
        clusters_authors = []
        for i in range(len(clusters)):
            if clusters[i] == submission_cluster:
                clusters_authors.append(row['authorPublicationHistory'][i])
                if row['authorPublicationHistory'][i]['doi'] == row['doi']:
                    same += 1
        
        embeddings_dataset.at[index, 'cluster_filter'] = clusters_authors

        #all_embeddings.append(embeddings)
        
        i += 1
        total += 1
    print("same", same, "total", total)
    
    embeddings_dataset.to_csv('data/embeddings_model_v3.csv', index=False)   
    

def clean_array_string(array_string):
    # Remove the 'array(' and ')' parts from the string
    array_string = re.sub(r'array\(', '', array_string)
    array_string = re.sub(r'\)', '', array_string)
    return array_string


def make_clusters(embeddings_dataset):
    all_embeddings = np.load('data/all_embeddings.npy', allow_pickle=True)
  
    same = 0
    total = 0
    embeddings_dataset['clusters'] = None
    print("Lentgh of Embeddings", len(all_embeddings))
    for index, row in tqdm.tqdm(embeddings_dataset.iterrows(), total=embeddings_dataset.shape[0]):
        s_id = row['SubmissionID']
        embeddings = all_embeddings[index]
        print("Length of Embeddings", len(embeddings))
        
        embeddings = np.array(embeddings)
        # Flatten the embeddings
        embeddings = np.array([np.array(embedding).flatten() for embedding in embeddings])
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        print(clusters)
        # print(len(clusters))
        # print(len(row['authorPublicationHistory']))
        # Get the last cluster
        last_cluster = clusters[0]
        clusters = clusters[1:]
        #print(row['doi'])
        clusters_authors = []
        for i in range(len(clusters)):
            if clusters[i] == last_cluster:
                #print("-----------------------------------------------", row['authorPublicationHistory'][i]['doi'])
                print("Length of Authors:", len(row['authorPublicationHistory']))
                print("Length of Clusters:", len(clusters))
                if len(row['authorPublicationHistory']) == 2:
                    print(row['authorPublicationHistory'])
                clusters_authors.append(row['authorPublicationHistory'][i])
                if row['authorPublicationHistory'][i]['doi'] == row['doi']:
                    same += 1   
            else:
                #print(row['authorPublicationHistory'][i]['doi'])
                continue
        
        embeddings_dataset.at[index, 'clusters'] = clusters_authors
        total += 1
    
    embeddings_dataset.to_csv('data/embeddings_model_v3.csv', index=False)
        
    print("same", same, "total", total)
def main():
    if not os.path.exists('data/embeddings_model_v3.csv') or True:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Check if GPU is available and move the model to GPU
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("GPU is available")
        else:
            print("GPU is not available")
        
        embeddings_dataset = pd.read_csv('data/embeddings_model.csv')
        create_clusters(embeddings_dataset, model)
    else:
        embeddings_dataset = pd.read_csv('data/embeddings_model_v2.csv')
        make_clusters(embeddings_dataset)
        
main()