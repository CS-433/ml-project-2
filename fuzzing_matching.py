from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
import csv
import tqdm
import json
import re
import ast
import os
import torch


def get_fuzzy_score(s1, s2):
    return fuzz.token_sort_ratio(s1, s2)

def fuzzy_matching(dataset):
    dataset['fuzzing_max_score'] = None
    for index, row in tqdm.tqdm(dataset.iterrows(), total=dataset.shape[0]):
        submissionTitle = row['SubmissionTitle']
        submissionAbstract = row['SubmissionAbstract']
        doi = row['doi']
        row['authorPublicationHistory'] = ast.literal_eval(row['authorPublicationHistory'])
        dataset.at[index, 'authorPublicationHistory'] = row['authorPublicationHistory']
        
        author_all_works_similarity = {}
        for author in row['authorPublicationHistory']:
            workTitle = author['title'] 
            workAbstract = author['abstract']
            workDOI = author['doi']
            
            submissionTitle_score = get_fuzzy_score(submissionTitle, workTitle)
            submissionAbstract_score = get_fuzzy_score(submissionAbstract, workAbstract)
            final_score = 0.3*submissionTitle_score + 0.7*submissionAbstract_score
            author_all_works_similarity[workDOI] = final_score
        
        # Get the work DOI with the highest similarity score
        max_score = max(author_all_works_similarity.keys(), key=(lambda k: author_all_works_similarity[k]))
        dataset.at[index, 'fuzzing_max_score'] = max_score
    
    dataset.to_csv('data/fuzzy_matching.csv', index=False)


def main():
    dataset = pd.read_csv('data/mini_dataset_v3.csv')
    # print(dataset.head())
    # print(dataset.columns)
    # print(dataset.shape)
    fuzzy_matching(dataset)
    
main()
    