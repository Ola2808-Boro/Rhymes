import glob
import requests
from bs4 import BeautifulSoup
import os
from pathlib import Path
from urllib.parse import quote
import csv
import random
from algorithm import Algorithm

def create_datasets(field,extended_dictionary):
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
                                             print('extended_dictionary')
                                             writer.writerow([original_word,checked_word,algorithm.get_score(original_word,checked_word)])
                                        else:
                                             
                                            writer.writerow([line.replace(',','').strip(),span.text.replace(',','').strip(),1])
                                        #print('For rhyme',line.replace(',','').strip(),span.text.replace(',','').strip())
                                    
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
                                                print('extended_dictionary')
                                                writer.writerow([original_word,checked_word,algorithm.get_score(original_word,checked_word)])
                                            else:
                                                writer.writerow([original_word,checked_word,0])
                                            #print('For no-rhyme',line.replace(',','').strip(),lines[word_index].replace(',','').strip())
                                        

                                         
                                    
            idx=idx+1
                         



def create_dataset_file(field,extended_dictionary):
    if extended_dictionary:
        #path='Rhymes/rhymes_dataset/extended_rhymes.csv'
        path='C:/Users/olkab/Desktop/Project Rhyme/Rhymes/rhymes_dataset/extended_rhymes.csv'
    else:
        path='Rhymes/rhymes_dataset/rhymes.csv'
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(field)


def expanding_dictionary(extended_dictionary:bool,field:str,basic_dictionary_name:str,extended_dictionary_name:str):
     algorithm=Algorithm()
     create_dataset_file(field=field,extended_dictionary=extended_dictionary)
     with open(basic_dictionary_name,'r',encoding="utf-8") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
             print(line)
             base_word, rhyme_candidate, score=line
             with open(extended_dictionary_name,'a',encoding="utf-8",newline='') as file:
                new_score=algorithm.get_score(base_word,rhyme_candidate)
                writer = csv.writer(file)
                writer.writerow([base_word,rhyme_candidate,new_score])
       
                     
                  
     
extended_dictionary_path='C:/Users/olkab/Desktop/Project Rhyme/Rhymes/rhymes_dataset/extended_rhymes_small_2.csv'
basic_dictionary_path='C:/Users/olkab/Desktop/Project Rhyme/Rhymes/rhymes_dataset/rhymes_small_2.csv'
field=["base_word", "rhyme_candidate","score"]
expanding_dictionary(extended_dictionary=True,field=field,basic_dictionary_name=basic_dictionary_path,extended_dictionary_name=extended_dictionary_path)

# create_datasets(field=extended_dictionary,extended_dictionary=True)