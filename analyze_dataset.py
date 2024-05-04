import numpy as np
import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def analyze_dataset(path_dataset:str):
    """
    Analyzes the dataset located at the specified path, categorizes scores into ranges, and visualizes the distribution.
    
    Args:
        path_dataset (str): The path to the dataset CSV file.
    """
    with open(path_dataset,'r',encoding="utf-8") as file:
        range_scores={
            '0.0-0.3':0,
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
            keys_selected.append(key)
        number_of_occurrences_of_the_score=[]
        for key in keys_selected:
            num=all_score.count(key)
            number_of_occurrences_of_the_score.append(num)
            all_score_obj.append({
                key:num
            })
            if 0.0<=float(key)<0.3:
                range_scores['0.0-0.3']+=num
            elif 0.3<=float(key)<0.4:
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
        
        counts_range=range_scores.items()
        plot_charts(all_score=all_score_obj,x=keys_selected,y=number_of_occurrences_of_the_score,range_score=counts_range)


def plot_charts(all_score:list,x:list,y:list,range_score:dict):
    """
    Plots charts based on the analyzed dataset and the distribution of scores.
    
    Args:
        all_score (list): List of dictionaries containing score occurrences.
        x (list): List of score ranges.
        y (list): List of the number of occurrences for each score range.
        range_score (dict): Dictionary containing score ranges and their respective counts.
    """
    df=pd.DataFrame(range_score, columns=['rhyming score','number of word pairs'])
    sns.set_theme(style="darkgrid")
    ax=sns.barplot(df, x='rhyming score',y='number of word pairs',hue='number of word pairs',legend=True,palette=sns.color_palette())
    plt.savefig("dataset_analysis.png")

path_dataset='C:/Users/olkab/Desktop/Project Rhyme/Rhymes/algorithm_dataset_all.csv'
analyze_dataset(path_dataset=path_dataset)

