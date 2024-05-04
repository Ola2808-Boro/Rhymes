import glob
import requests
from bs4 import BeautifulSoup
import os
from pathlib import Path
from urllib.parse import quote
import csv
import random
from algorithm import Algorithm
import random
import sys
sys.path.append('C:/Users/olkab/Desktop/Project Rhyme/Rhymes/faster_algorithm')
from faster_algorithm  import get_scoreboard,get_score
import pandas as pd
import json
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten,Dropout
from keras.preprocessing.text import Tokenizer
import glob
from sklearn.metrics import r2_score,mean_squared_error
from keras.preprocessing.sequence import pad_sequences
import numpy as np


BASE_DIR='C:/Users/olkab/Desktop/Project Rhyme/Rhymes'
max_sequence_length = 20
tokenizer = Tokenizer(char_level=True, filters=None, lower=True)
polish_alphabet = 'aąbcćdeęfghijklłmnńoópqrsśtuvwyzźż'
tokenizer.fit_on_texts(polish_alphabet)
embedding_dim = 8 
vocab_size = len(tokenizer.word_index) + 1



Model1 = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=2*max_sequence_length),
    Flatten(),
    # LSTM(units=64),
    Dense(units=64, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=1, activation='sigmoid')  # Output layer for binary classification
])

Model2=Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=2*max_sequence_length),
            Flatten(),
            Dense(464,activation='relu'),
            Dense(50,activation='relu'),
            Dropout(0.2),
            Dense(25,activation='relu'),
            Dropout(0.2),
            Dense(16,activation='relu'),
            Dropout(0.2),
            Dense(8,activation='relu'),
            Dropout(0.2),
            Dense(1,activation='sigmoid'),
])

Model3 = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=2*max_sequence_length),
    LSTM(units=64),
    Dense(units=64, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=1, activation='sigmoid')  # Output layer for binary classification
])

def create_datasets_algorithm(field,extended_dictionary):
    """
    Generates a dataset for the algorithm by selecting rhymes from an extended dictionary.
    
    Args:
        field (list): A list containing field names for the dataset.
        extended_dictionary (str): Path to the extended dictionary CSV file.
    """
    data=pd.read_csv(extended_dictionary)
    no_rhymes=data.where(data['score']==0).dropna(how='all')[5000:15000]
    rhymes=data.where(data['score']!=0).dropna(how='all')[15000:20000]
    with open(f'C:/Users/olkab/Desktop/Project Rhyme/Rhymes/algorithm_dataset_3.csv','a', encoding="utf-8",newline='') as f_name:
        field=["base_word", "rhyme_candidate","score"]
        writer = csv.writer(f_name)
        writer.writerow(field)
        for idx in rhymes.index:
            if len(get_scoreboard(rhymes["rhyme_candidate"][idx]))>2:
                num_results=len(get_scoreboard(rhymes["rhyme_candidate"][idx]))
                indexes=random.sample(range(0,num_results-1),2)
                for index in indexes:
                    (word,score)=get_scoreboard(rhymes["rhyme_candidate"][idx])[index]
                    #print(word)
                    base_word=rhymes["rhyme_candidate"][idx]
                    rhyme_candidate=word
                    result=score
                    writer.writerow([base_word,rhyme_candidate,result])
        no_rhymes.to_csv('C:/Users/olkab/Desktop/Project Rhyme/Rhymes/algorithm_dataset_2.csv', mode='a', index=False, header=False)
    print(data)

def create_datasets(field,extended_dictionary):

    """
    Creates datasets for both rhyming and non-rhyming words, considering an extended dictionary if specified.
    
    Args:
        field (list): A string containing the field names for the dataset.
        extended_dictionary (bool): Indicates whether to use an extended dictionary or not.
    """
    algorithm=Algorithm()
    idx=0
    create_dataset_file(field,extended_dictionary)
    files=glob.glob('C:/Users/olkab/Desktop/Project Rhyme/Rhymes/dictionary_no_space/*.txt')
    for file in files:
        if file.endswith(".txt"):
            print(file)
            with open(f'{file}','r',encoding="utf-8") as f:
                            lines=f.readlines()
                            for line in lines:
                                print(line)
                                page_url = 'https://polskierymy.pl/?rymy='+quote(line.replace('\n',''))
                                print(page_url)
                                page=requests.get(page_url)
                                soup=BeautifulSoup(page.content,'html.parser')
                                spans=soup.find_all('span',{'class':'result'})
                                with open(f'C:/Users/olkab/Desktop/Project Rhyme/Rhymes/rhymes_dataset/extended_rhymes.csv','a', encoding="utf-8",newline='') as f_name:
                                    all_rhymes=spans
                                    dictionaries_idx=random.sample(range(0, len(files)-1), 2)
                                    for span in spans[1:3]:
                                        original_word=line.replace(',','').strip()
                                        checked_word=span.text.replace(',','').strip()
                                        writer = csv.writer(f_name)
                                        if extended_dictionary:
                                             
                                            writer.writerow([original_word,checked_word,algorithm.get_score(original_word,checked_word)])
                                        else:
                                             
                                            writer.writerow([line.replace(',','').strip(),span.text.replace(',','').strip(),1])
                                    
                                    for idx in dictionaries_idx:
                                        dictionary=files[idx]
                                        with open(f'{dictionary}','r',encoding="utf-8") as file_name:
                                            lines= file_name.readlines()
                                        
                                            word_index=random.randint(0,len(lines)-1)
                                            while lines[word_index] in all_rhymes:
                                                word_index=random.randint(0,len(lines)-1)
                                            original_word=line.replace(',','').strip()
                                            checked_word=span.text.replace(',','').strip()
                                            if extended_dictionary:
                                                writer.writerow([original_word,checked_word,algorithm.get_score(original_word,checked_word)])
                                            else:
                                                writer.writerow([original_word,checked_word,0])
                                            
                                        

                         
                                    
            idx=idx+1
                         


def predict_func(word_1:str, word_2:str,model):
    """
    Predicts a score for two words using a given model.
    
    Args:
        word_1 (str): The first word.
        word_2 (str): The second word.
        model: The trained model for prediction.
    
    Returns:
        raw_result: The raw prediction result.
    """
    seq_word1 = tokenizer.texts_to_sequences([word_1])
    seq_word2 = tokenizer.texts_to_sequences([word_2])

    max_sequence_length = 20
    pad_word1 = pad_sequences(seq_word1, maxlen=max_sequence_length)
    pad_word2 = pad_sequences(seq_word2, maxlen=max_sequence_length)

    x = np.concatenate((pad_word1, pad_word2), axis=-1)
    raw_result=model.predict(x)[0]
    return raw_result

def create_dataset_file(field,extended_dictionary,extended_dictionary_name):
    """
    Creates a CSV file for storing the dataset, including the specified field names.
    
    Args:
        field (list): A string containing the field names for the dataset.
        extended_dictionary (bool): Indicates whether to use an extended dictionary or not.
        extended_dictionary_name (str): The name of the extended dictionary CSV file.
    """
    if extended_dictionary:
        path=extended_dictionary_name
    else:
        path='Rhymes/rhymes_dataset/rhymes.csv'
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(field)


def expanding_dictionary(extended_dictionary:bool,field:list,basic_dictionary_name:str,extended_dictionary_name:str):
    """
    Expands the dictionary by adding rhymes and their scores to an extended dictionary file.
    
    Args:
        extended_dictionary (bool): Indicates whether to use an extended dictionary or not.
        field (list): A string containing the field names for the dataset.
        basic_dictionary_name (str): Path to the basic dictionary CSV file.
        extended_dictionary_name (str): The name of the extended dictionary CSV file.
    """
     #slower algorithm
     #algorithm=Algorithm() 
    create_dataset_file(field=field,extended_dictionary=extended_dictionary,extended_dictionary_name=extended_dictionary_name)
    with open(basic_dictionary_name,'r',encoding="utf-8") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
             base_word, rhyme_candidate, score=line
             with open(extended_dictionary_name,'a',encoding="utf-8",newline='') as file:
                new_score=get_score(base_word,rhyme_candidate)
                if new_score!=-1:
                    writer = csv.writer(file)
                    writer.writerow([base_word,rhyme_candidate,new_score])
       
                   
def read_file(file_path:str):

    """
    Reads a CSV file and returns its content.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        csv: The content of the CSV file.
    """

    csv=pd.read_csv(file_path)
    print(csv.columns)
    return csv


def metrics(result_file_path:str):
    """
    Calculates evaluation metrics and saves them to a JSON file.
    
    Args:
        result_file_path (str): Path to the CSV file containing model results.
    """
     
    results=pd.read_csv(result_file_path)
    model_1_score=results["model_1_score"].values
    model_2_score=results["model_2_score"].values
    algorithm_score=results["algorithm_score"].values
    MSE_model_1_algorithm=mean_squared_error(algorithm_score,model_1_score)
    MSE_model_2_algorithm=mean_squared_error(algorithm_score,model_2_score)
    R2_model_1_algorithm=r2_score(algorithm_score,model_1_score)
    R2_model_2_algorithm=r2_score(algorithm_score,model_2_score)
    metrics_data={
        'MSE_model_1_algorithm':MSE_model_1_algorithm,
        'MSE_model_2_algorithm':MSE_model_2_algorithm,
        'R2_model_1_algorithm':R2_model_1_algorithm,
        'R2_model_2_algorithm':R2_model_2_algorithm,
    }
    json_object = json.dumps(metrics_data, indent=len(metrics_data))
    with open(f"{BASE_DIR}/metrics.json", "w") as outfile:
        outfile.write(json_object)


def results(models_path:list,words:pd.DataFrame):

    """
    Generates model results for a given set of words and saves them to a CSV file.
    
    Args:
        models_path (list): A list containing paths to the model weights.
        words (pd.DataFrame): DataFrame containing words for evaluation.
    """

    field=["base_word", 
           "rhyme_candidate",
           "score",
           "model_1_score",
           "model_2_score",
           "algorithm_score"]
    
    extended_dictionary_name=f'{BASE_DIR}/results.csv'
    create_dataset_file(field=field,extended_dictionary=True,extended_dictionary_name=extended_dictionary_name)
    model1=Model1
    model2=Model2
    model1.load_weights(models_path[0])
    model2.load_weights(models_path[1])
    with open(extended_dictionary_name,'a',encoding="utf-8",newline='') as file:
        for idx in range(len(words)):
            if idx!=0:
                base_word=words['base_word'][idx]
                rhyme_candidate=words['rhyme_candidate'][idx]
                score=words['score'][idx]
                algorithm_score=get_score(base_word,rhyme_candidate)
                model_1_raw_result=predict_func(word_1=base_word, word_2=rhyme_candidate,model=model1)[0]
                model_2_raw_result=predict_func(word_1=base_word, word_2=rhyme_candidate,model=model2)[0]
                writer = csv.writer(file)
                print(base_word,rhyme_candidate,score,model_1_raw_result,model_2_raw_result,algorithm_score)
                writer.writerow([base_word,
                                rhyme_candidate,
                                score,
                                model_1_raw_result,
                                model_2_raw_result,
                                algorithm_score
                                ])
        metrics(f'{BASE_DIR}/results.csv')
        


path_dataset=f'{BASE_DIR}/rhymes_dataset/extended_rhymes_small_3_faster_algorithm.csv' 
extended_dictionary_path='C:/Users/olkab/Desktop/Project Rhyme/Rhymes/rhymes_dataset/extended_rhymes_small_3_faster_algorithm.csv'
basic_dictionary_path='C:/Users/olkab/Desktop/Project Rhyme/Rhymes/rhymes_dataset/rhymes_small_3.csv'
field=["base_word", "rhyme_candidate","score"]
#expanding_dictionary(extended_dictionary=True,field=field,basic_dictionary_name=basic_dictionary_path,extended_dictionary_name=extended_dictionary_path)
path_dataset=f'{BASE_DIR}/rhymes_dataset/extended_rhymes_small_3_faster_algorithm.csv' 
path=f'{BASE_DIR}/rhymes_dataset/rhymes_17500x40.csv'

model1_weights_path=f'C:/Users/olkab/Desktop/Project Rhyme/Rhymes/models/model1/model1_30_32_SGD.weights.best.hdf5'
model2_weights_path=f'C:/Users/olkab/Desktop/Project Rhyme/Rhymes/models/model2/model2_30_32_SGD.weights.best.hdf5'
# model3_weights_path=f'{BASE_DIR}/models/model3/model3_10_64_Adam.weights.best.hdf5'
models_path=[model1_weights_path,model2_weights_path]
#create_datasets_algorithm(field,extended_dictionary='C:/Users/olkab/Desktop/Project Rhyme/Rhymes/rhymes_dataset/rhymes_small_2.csv')
results(models_path=models_path,words=read_file(path_dataset))
metrics(f'{BASE_DIR}/results.csv')
# create_datasets(field=extended_dictionary,extended_dictionary=True)