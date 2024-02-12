import numpy as np
import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def analyze_dataset(path_dataset:str):

    with open(path_dataset,'r',encoding="utf-8") as file:
        range_scores={
            # '0.1-0.2':0,
            # '0.2-0.3':0,
            '0.3-0.4':0,
            '0.4-0.5':0,
            '0.5-0.6':0,
            '0.6-0.7':0,
            '0.7-0.8':0,
            '0.8-0.9':0,
            '0.9-1.0':0,
        }
        csvreader = csv.reader(file)
        all_score=[]
        all_score_obj=[]
        keys_selected=[]
        for line in csvreader:
            base_word,rhyme_candidate,score=line
            all_score.append(score)
        keys=np.unique(all_score)[:-1]
        for key in keys:
            if float(key)>=0.3:
                keys_selected.append(key)
        number_of_occurrences_of_the_score=[]
        for key in keys_selected:
            #print(key)
            num=all_score.count(key)
            number_of_occurrences_of_the_score.append(num)
            all_score_obj.append({
                key:num
            })
            if 0.3<=float(key)<0.4:
                range_scores['0.3-0.4']+=num
            elif 0.4<=float(key)<0.5:
                range_scores['0.4-0.5']+=num
            elif 0.5<=float(key)<0.6:
                range_scores['0.5-0.6']+=num
            elif 0.6<=float(key)<0.7:
                range_scores['0.6-0.7']+=num
            elif 0.7<=float(key)<0.8:
                range_scores['0.7-0.8']+=num
            elif 0.8<=float(key)<0.9:
                range_scores['0.8-0.9']+=num
            elif 0.9<=float(key)<1.0:
                range_scores['0.9-1.0']+=num
        
        #print(all_score_obj)
        counts_range=range_scores.items()
        print('Trick', counts_range)
        plot_charts(all_score=all_score_obj,x=keys_selected,y=number_of_occurrences_of_the_score,range_score=counts_range)


def plot_charts(all_score:list,x:list,y:list,range_score:dict):
    print(len(x),len(y),x,y)
    df=pd.DataFrame(range_score, columns=['rhyming score','number of word pairs'])
    #df.insert(2, "range_array", [21, 23, 24, 21], True)
    print(df.head())
    sns.set_theme(style="darkgrid")
    ax=sns.barplot(df, x='rhyming score',y='number of word pairs',hue='number of word pairs',legend=True,palette=sns.color_palette())
    # print(ax.containers)
    # ax.bar_label(ax.containers, fontsize=10);
    plt.savefig("dataset_analysis.png")

path_dataset='C:/Users/olkab/Desktop/Project Rhyme/Rhymes/rhymes_dataset/extended_rhymes_17500x40.csv'     
analyze_dataset(path_dataset=path_dataset)
# tips = sns.load_dataset("tips")
# print(tips)