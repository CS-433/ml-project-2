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

def preprocessing_text(dataset):
    # Lower Case the Titles
    for index, row in tqdm.tqdm(dataset.iterrows(), total=dataset.shape[0]):
        remove_index = None
        
        row['SubmissionTitle'] = row['SubmissionTitle'].lower()
        row['SubmissionAbstract'] = row['SubmissionAbstract'].lower()
        
        # Use RE to remove special characters and punctuations
        row['SubmissionTitle'] = re.sub(r'[^\w\s]', '', row['SubmissionTitle'])
        row['SubmissionAbstract'] = re.sub(r'[^\w\s]', '', row['SubmissionAbstract'])
        
        
        # Update the dataset
        dataset.at[index, 'SubmissionTitle'] = row['SubmissionTitle']
        dataset.at[index, 'SubmissionAbstract'] = row['SubmissionAbstract']
    
        
        row['authorPublicationHistory'] = ast.literal_eval(row['authorPublicationHistory'])
        dataset.at[index, 'authorPublicationHistory'] = row['authorPublicationHistory']
        
        
        # print(row['authorPublicationHistory'][0]['title'])	
        # exit()
        
        # try:
        #     row['authorPublicationHistory'] = json.loads(row['authorPublicationHistory'])
        # except json.JSONDecodeError as e:
        #     print(f"Error decoding JSON for index {index}: {e}")
        #     print(f"Original string: {row['authorPublicationHistory']}")
        #     row['authorPublicationHistory'] = json.loads(row['authorPublicationHistory'].replace("'", "\"").replace("None", "null").replace("\\", "\\\\"))
        i = 0
        for authorWorks in row['authorPublicationHistory']:
            if authorWorks['title'] == None or authorWorks['abstract'] == None:
                remove_index = row['authorPublicationHistory'].index(authorWorks)
                break
            authorWorks['title'] = authorWorks['title'].lower()
            authorWorks['abstract'] = authorWorks['abstract'].lower()
            authorWorks['title'] = re.sub(r'[^\w\s]', '', authorWorks['title'])
            authorWorks['abstract'] = re.sub(r'[^\w\s]', '', authorWorks['abstract'])
            
            dataset.at[index, 'authorPublicationHistory'][i] = authorWorks
            i += 1
            
        if remove_index != None:
            # Remove that whole row from the dataset
            dataset = dataset.drop(index)
        else:

            # Update the dataset with the modified authorPublicationHistory
            dataset.at[index, 'authorPublicationHistory'] = row['authorPublicationHistory']
    return dataset

def preprocessing_authors(dataset):
    # Make all the firstName and lastName lowercase
    for index, row in tqdm.tqdm(dataset.iterrows(), total=dataset.shape[0]):
        row['firstName'] = row['firstName'].lower()
        row['lastName'] = row['lastName'].lower()
        
        # Update the dataset
        dataset.at[index, 'firstName'] = row['firstName']
        dataset.at[index, 'lastName'] = row['lastName']
        
        if row['middleName'] == None or type(row['middleName']) == float:
            row['middleName'] = None
        else:
            row['middleName'] = row['middleName'].lower()
            # Covert "J. K." as "JK"
            row['middleName'] = row['middleName'].replace(". ", "")
        
        
        if row['middleName'] == None:
            fullName = row['firstName'] + " " + row['lastName']
        else:
            fullName = row['firstName'] + " " + row['middleName'] + " " + row['lastName']
        
        
        # Update the dataset with fullName
        dataset.at[index, 'fullName'] = fullName
        
        for i in range(len(row['authorPublicationHistory'])):
            authors = row['authorPublicationHistory'][i]['authors']
            for j in range(len(authors)):
                authors[j] = authors[j].lower()
                authors[j] = authors[j].replace(". ", "")
                dataset.at[index, 'authorPublicationHistory'][i]['authors'] = authors
                
    return dataset

def generate_text_embeddings(dataset, model):
    dataset['title_embedding'] = None
    dataset['abstract_embedding'] = None
    dataset['doi_embedding'] = None
    dataset['authorPublicationHistory_embedding'] = None
    for index, row in tqdm.tqdm(dataset.iterrows(), total=dataset.shape[0]):
        # Generate the embeddings for the title
        title_embedding = model.encode(row['SubmissionTitle'], convert_to_tensor=True)
        dataset.at[index, 'title_embedding'] = title_embedding.cpu().numpy().tolist()
        break_point = False
        
        # Generate the embeddings for the abstract
        abstract_embedding = model.encode(row['SubmissionAbstract'], convert_to_tensor=True)
        dataset.at[index, 'abstract_embedding'] = abstract_embedding.cpu().numpy().tolist()
        
        # Generate the embeddings for the DOI
        doi_embedding = model.encode(row['SubmissionID'], convert_to_tensor=True)
        dataset.at[index, 'doi_embedding'] = doi_embedding.cpu().numpy().tolist()
        
        # Generate the embeddings for the authorPublicationHistory
        authorPublicationHistory_embedding = []
        # row['authorPublicationHistory'] = ast.literal_eval(row['authorPublicationHistory'])
        # dataset.at[index, 'authorPublicationHistory'] = row['authorPublicationHistory']
        for authorWorks in row['authorPublicationHistory']:
            authorTitleEmbedding = model.encode(authorWorks['title'], convert_to_tensor=True)
            authorAbstractEmbedding = model.encode(authorWorks['abstract'], convert_to_tensor=True)
            if authorWorks['doi'] == None:
                break_point = True
            else:
                authorDOIEmbedding = model.encode(authorWorks['doi'], convert_to_tensor=True)
                authorPublicationHistory_embedding.append({
                    'title_embedding': authorTitleEmbedding.cpu().numpy().tolist(),
                    'abstract_embedding': authorAbstractEmbedding.cpu().numpy().tolist(),
                    'doi_embedding': authorDOIEmbedding.cpu().numpy().tolist()
                })
        
        if break_point:
            # Remove that whole row from the dataset
            dataset = dataset.drop(index)
        else:
            # Update the dataset with the modified authorPublicationHistory
            dataset.at[index, 'authorPublicationHistory_embedding'] = authorPublicationHistory_embedding
    
    return dataset
    
    

def main():
    if os.path.exists('data/preprocessed_dataset.csv'):
        dataset = pd.read_csv('data/preprocessed_dataset.csv')
    else:
        dataset = pd.read_csv('data/extended_dataset_v3.csv')
        dataset = preprocessing_text(dataset)
        print(dataset.head())
        print(dataset.shape)
        
        dataset = preprocessing_authors(dataset)
        print(dataset.head())
        print(dataset.shape)
        
        # Safe the preprocessed dataset
        dataset.to_csv('data/preprocessed_dataset.csv', index=False)
    
    
    model = SentenceTransformer('all-mpnet-base-v2')
    # Check if GPU is available and move the model to GPU
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("GPU is available")
    else:
        print("GPU is not available")
    embeddings_model = generate_text_embeddings(dataset, model)
    embeddings_model.to_csv('data/embeddings_model.csv', index=False)
    print(embeddings_model.head())
    print(embeddings_model.shape)
main()
    