import pandas as pd
import csv
import tqdm
import json
import re
import ast
import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def main():
    dataset = pd.read_csv('data/predicted_published_work.csv')
    
    y_true = dataset['doi']
    y_pred = dataset['predicted_published_work']
    
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
            
main()