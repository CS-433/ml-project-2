import pandas as pd
import csv
import tqdm
import json
import re
import ast
import os
import torch

from sentence_transformers import SentenceTransformer


def predict_published_words(dataset, model):
    dataset['predicted_published_work'] = None
    for index, row in tqdm.tqdm(dataset.iterrows(), total=dataset.shape[0]):
        title_embedding = model.encode(row['SubmissionTitle'])
        abstract_embedding = model.encode(row['SubmissionAbstract'])
        doi = row['doi']
        row['tfidf_cosine_similarity'] = ast.literal_eval(row['tfidf_cosine_similarity'])
        dataset.at[index, 'tfidf_cosine_similarity'] = row['tfidf_cosine_similarity']
        
        author_all_works_similarity = {}
        for authorWorks in row['tfidf_cosine_similarity']:
            authorTitleEmbedding = model.encode(authorWorks['title'])
            authorAbstractEmbedding = model.encode(authorWorks['abstract'])
            
            # Convert NumPy arrays to PyTorch tensors and add an extra dimension
            title_embedding_tensor = torch.tensor(title_embedding).unsqueeze(0)
            authorTitleEmbedding_tensor = torch.tensor(authorTitleEmbedding).unsqueeze(0)
            abstract_embedding_tensor = torch.tensor(abstract_embedding).unsqueeze(0)
            authorAbstractEmbedding_tensor = torch.tensor(authorAbstractEmbedding).unsqueeze(0)
            
            # Compute the cosine similarity
            title_cosine_similarity = torch.nn.functional.cosine_similarity(title_embedding_tensor, authorTitleEmbedding_tensor)
            abstract_cosine_similarity = torch.nn.functional.cosine_similarity(abstract_embedding_tensor, authorAbstractEmbedding_tensor)

            author_all_works_similarity[authorWorks['doi']] = 0.3 * title_cosine_similarity.item() + 0.7 * abstract_cosine_similarity.item()
            
        # Sort the list of cosine similarities
        author_all_works_similarity = dict(sorted(author_all_works_similarity.items(), key=lambda item: item[1], reverse=True))
        # Get the top most similar work
        top_work = list(author_all_works_similarity.items())[0]
        dataset.at[index, 'predicted_published_work'] = top_work[0]
    
    dataset.to_csv('data/predicted_published_work.csv', index=False)


def main():
    model = SentenceTransformer('all-mpnet-base-v2')
    # Check if GPU is available and move the model to GPU
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("GPU is available")
    else:
        print("GPU is not available")
        
        
    dataset = pd.read_csv('data/tfidf_cosine_similarity.csv')
    
    predict_published_words(dataset, model)

main()