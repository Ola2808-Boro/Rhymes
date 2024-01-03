import glob
import requests
from bs4 import BeautifulSoup
import os
from pathlib import Path
from urllib.parse import quote
import csv
import random


def create_datasets():
    idx=0
    #create_dataset_file()
    files=glob.glob('Rhymes/dictionary_no_space/*.txt')
    for file in files:
        if file.endswith(".txt"):
            #print(file)
            with open(f'{file}','r',encoding="utf-8") as f:
                            lines=f.readlines()
                            for line in lines:
                                print(line)
                                page_url = 'https://polskierymy.pl/?rymy='+quote(line.replace('\n',''))
                                print(page_url)
                                page=requests.get(page_url)
                                soup=BeautifulSoup(page.content,'html.parser')
                                spans=soup.find_all('span',{'class':'result'})
                                with open(f'Rhymes/dataset/rhymes.csv','a', encoding="utf-8",newline='') as f_name:
                                    all_rhymes=spans
                                    dictionaries_idx=random.sample(range(0, len(files)-1), 2)
                                    for span in spans[1:3]:
                                        writer = csv.writer(f_name)
                                        writer.writerow([line.replace(',','').strip(),span.text.replace(',','').strip(),1])
                                        #print('For rhyme',line.replace(',','').strip(),span.text.replace(',','').strip())
                                    
                                    for idx in dictionaries_idx:
                                        dictionary=files[idx]
                                        with open(f'{dictionary}','r',encoding="utf-8") as file_name:
                                            lines= file_name.readlines()
                                        
                                            word_index=random.randint(0,len(lines)-1)
                                            while lines[word_index] in all_rhymes:
                                                word_index=random.randint(0,len(lines)-1)
                                            writer.writerow([line.replace(',','').strip(),lines[word_index].replace(',','').strip(),0])
                                            #print('For no-rhyme',line.replace(',','').strip(),lines[word_index].replace(',','').strip())
                                        

                                         
                                    
            idx=idx+1
                         



def create_dataset_file():
    with open('Rhymes/dataset/rhymes.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["base_word", "rhyme_candidate", "score"]
        writer.writerow(field)




create_datasets()