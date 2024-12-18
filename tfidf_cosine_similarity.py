from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import csv
import tqdm
import json
import re
import ast
import os
import torch

def compute_tfidf_cosine_similarity(dataset):
    # Compute the TF-IDF matrix
    vectorizer = TfidfVectorizer()
    dataset['tfidf_cosine_similarity'] = None
    # Combine all texts for fitting the vectorizer
    all_texts = []
    for index, row in tqdm.tqdm(dataset.iterrows(), total=dataset.shape[0]):
        all_texts.append(row['SubmissionTitle'])
        all_texts.append(row['SubmissionAbstract'])
        row['cluster_filter'] = ast.literal_eval(row['cluster_filter'])
        for authorWorks in row['cluster_filter']:
            all_texts.append(authorWorks['title'])
            all_texts.append(authorWorks['abstract'])
    
    vectorizer.fit(all_texts)
    
    for index, row in tqdm.tqdm(dataset.iterrows(), total=dataset.shape[0]):
        all_cosine_similarities = {}
        doi = row['doi']
        # Transform the title and abstract
        titletfidf = vectorizer.transform([row['SubmissionTitle']])
        abstracttfidf = vectorizer.transform([row['SubmissionAbstract']])
        
        row['cluster_filter'] = ast.literal_eval(row['cluster_filter'])
        dataset.at[index, 'cluster_filter'] = row['cluster_filter']
        
        for authorWorks in row['cluster_filter']:
            authDOI = authorWorks['doi']
            authorTitleTfidf = vectorizer.transform([authorWorks['title']])
            authorAbstractTfidf = vectorizer.transform([authorWorks['abstract']])
            
            # Compute the cosine similarity
            title_cosine_similarity = cosine_similarity(titletfidf, authorTitleTfidf)
            abstract_cosine_similarity = cosine_similarity(abstracttfidf, authorAbstractTfidf)
            
            all_cosine_similarities[authDOI] = 0.3 * title_cosine_similarity + 0.7 * abstract_cosine_similarity
    
        # Keep the top k most similar works
        if len(all_cosine_similarities) == 1:
            k = 1
        elif len(all_cosine_similarities) < 4:
            k = 2
        else:
            k = 3
        
        # Sort the list of cosine similarities
        all_cosine_similarities = dict(sorted(all_cosine_similarities.items(), key=lambda item: item[1], reverse=True))
        # Get the top k most similar works
        top_k = dict(list(all_cosine_similarities.items())[:k])
        
        # get the auth objects of the top k
        top_k_auths = []
        for key in top_k:
            for auth in row['cluster_filter']:
                if auth['doi'] == key:
                    top_k_auths.append(auth)
                    break
        
        dataset.at[index, 'tfidf_cosine_similarity'] = top_k_auths
    dataset.to_csv('data/tfidf_cosine_similarity.csv', index=False)

def main():
    dataset = pd.read_csv('data/embeddings_model_v3.csv')
    compute_tfidf_cosine_similarity(dataset)

main() 