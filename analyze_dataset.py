import numpy as np
import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def analyze_dataset(path_dataset:str):
    with open(path_dataset,'r',encoding="utf-8") as file:
        csvreader = csv.reader(file)
        all_score=[]
        all_score_obj=[]
        for line in csvreader:
            base_word,rhyme_candidate,score=line
            all_score.append(score)
        keys=np.unique(all_score)[:-1]
        counts=[]
        #print(keys)
        for key in keys:
            #print(key)
            num=all_score.count(key)
            counts.append(num)
            all_score_obj.append({
                key:num
            })
        #print(all_score_obj)
        plot_charts(all_score=all_score_obj,x=keys,y=counts)


def plot_charts(all_score:list,x:list,y:list):
    print(len(x),len(y),x,y)
    df=pd.DataFrame({
        'counts':y[:10],
        'score':x[:10]
        
    })
    print(df.head())
    sns.set_theme(style="darkgrid")
    sns.histplot(df,x='counts')
    plt.savefig("seaborn_plot.png")

path_dataset='C:/Users/olkab/Desktop/Project Rhyme/Rhymes/rhymes_dataset/extended_rhymes_17500x40.csv'     
analyze_dataset(path_dataset=path_dataset)
# tips = sns.load_dataset("tips")
# print(tips)