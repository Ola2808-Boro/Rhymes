import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote
from pathlib import Path
import os


"""
A function for removing empty lines from files containing words in Polish.
"""
def clean_files():
    if Path('Rhymes/dictionary_no_space').is_dir():
        print('Dictionary folder already exists ')
    else:
        os.mkdir('Rhymes/dictionary_no_space')
        print('Creating folder..')

    for file in os.listdir("Rhymes/dictionary"):
        if file.endswith(".txt"):
            with open(f'Rhymes/dictionary/{file}','r',encoding="utf-8") as f:
                lines=f.readlines()
                for line in lines:
                    if line.strip():
                        with open(f'Rhymes/dictionary_no_space/{file}.txt','a',encoding="utf-8") as new_file:
                            new_file.write(line)

"""
A function to create a txt file for each word in Polish, which contains rhymes for a given word.
"""
def create_rhymes_dictionary(all_letters):
    idx=0
    for file in os.listdir("Rhymes/dictionary_no_space/dictionary_no_space"):
        print(file)
        if file.endswith(".txt"):
            print(file)
            with open(f'Rhymes/dictionary_no_space/{file}','r',encoding="utf-8") as f:
                            lines=f.readlines()
                            for line in lines:
                                print(line)
                                page_url = 'https://polskierymy.pl/?rymy='+quote(line.replace('\n',''))
                                print(page_url)
                                page=requests.get(page_url)
                                soup=BeautifulSoup(page.content,'html.parser')
                                spans=soup.find_all('span',{'class':'result'})
                                if Path(f'Rhymes/rhymes/{all_letters[idx]}').is_dir():
                                    print('Yes')
                                else:
                                    os.mkdir(Path(f'Rhymes/rhymes/{all_letters[idx]}'))
                                    print('Yes')
                                with open(f'Rhymes/rhymes/{all_letters[idx]}/{line.strip()}_rhyme.txt','a', encoding="utf-8") as f:
                                    for span in spans:
                                        f.write(span.text+'\n')
            idx=idx+1
                         



"""
A function that creates txt files with all possible words in Polish. 
Then it calls the clean_files() function, which removes e.g. empty lines from these files. 
After this stage, it calls function create_rhymes_dictionary() to create a file for each word containing words that rhyme with it.
"""
def create_base_dictionary():
    if Path('Rhymes/dictionary').is_dir():
        print('Dictionary folder already exists ')
    else:
        os.mkdir('Rhymes/dictionary')
        print('Creating folder..')
    page_url = 'https://polski-slownik.pl/wszystkie-slowa-jezyka-polskiego.php'
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    list_all_buttons=soup.find_all('button',{'class':'btn-form'})
    all_letters=[]
    for button in list_all_buttons:
        all_letters.append(button.get_text()[-1].lower())
    print(f'All letter {all_letters}')
    numbers_of_letters=[]
    for letter in all_letters:
        page_url = 'https://polski-slownik.pl/wszystkie-slowa-jezyka-polskiego.php?id=na-litere-'+quote(letter)
        page=requests.get(page_url)
        soup=BeautifulSoup(page.content,'html.parser')
        buttons=soup.find_all('button',{'class':'btn-form'})
        for button in buttons:
            numbers_of_letters.append(re.findall('\d+',button.get_text())[0])
        print(f'Number of syllables {numbers_of_letters}')
        numbers_of_letters=list(dict.fromkeys(numbers_of_letters))
        for number in numbers_of_letters:
            page_url = 'https://polski-slownik.pl/wszystkie-slowa-jezyka-polskiego.php?id='+number+'-literowe-na-litere-'+quote(letter)
            page=requests.get(page_url)
            soup=BeautifulSoup(page.content,'html.parser')
            if soup.find_all('input',value="POKAŻ WSZYSTKIE SŁOWA"):
                page_url = 'https://polski-slownik.pl/wszystkie-slowa-jezyka-polskiego.php?id='+number+'-literowe-na-litere-'+quote(letter)+'&Submit=POKA%C5%BB+WSZYSTKIE+S%C5%81OWA'
                print(page_url)
                page=requests.get(page_url)
                soup=BeautifulSoup(page.content,'html.parser')
                words=soup.find('p',{'class':['mbr-text mbr-fonts-style', 'align-left',' display-7']}).getText().replace(',','').replace('.','').strip()
                with open(f'Rhymes/dictionary/dictionary_{letter}.txt','a',encoding="utf-8") as f:
                    try:
                        f.write(words.strip())
                    except:
                        pass
            else:
                paragraphs=soup.find_all('p', itemprop="name")
                with open(f'Rhymes/dictionary/dictionary_{letter}.txt','a',encoding="utf-8") as f:
                    for p in paragraphs:
                        f.write(p.get_text().replace(" ",""))
    
    clean_files()
    #create_rhymes_dictionary(all_letters=all_letters)
    numbers_of_letters=[]


create_base_dictionary()


