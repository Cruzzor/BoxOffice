#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import ast  # Interpret string as Python command
from typing import Tuple

def extract_names(keyword_list):
    if pd.isna(keyword_list):
        return []
    else:
        return list(map(lambda x: x['name'].replace(' ', '_'), ast.literal_eval(keyword_list)))
    

def has_top_keyword(keyword_list, top_keywords):
    if not pd.isna(keyword_list):
        list_of_keywords = extract_names(keyword_list)
        for keyword in list_of_keywords:
            if keyword in top_keywords:
                return True
    return False


def extract_has_top_keyword(data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    # Get pairs of keyword list + revenue for each movie
    df = data.copy()
    df['Keywords'] = df['Keywords'].map(extract_names, na_action=None)
    df = df[['Keywords', 'revenue']]

    keywords_df = (
        pd.DataFrame(df.Keywords.values.tolist())
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame('Keywords')
    )
    df = keywords_df.join(df[['revenue']]).reset_index(drop=True)

    # Compute sum and mean revenue for each keyword + count occurences
    def f(x):
        d = {}
        d['revenue_sum'] = x['revenue'].sum()
        d['revenue_mean'] = x['revenue'].mean()
        d['keyword_count'] = len(x['revenue'])
        return pd.Series(d, index=['revenue_sum', 'revenue_mean', 'keyword_count'])

    df = (df
        .groupby('Keywords')
        .apply(f)
        .sort_values(['revenue_sum', 'revenue_mean'], ascending=False)
        .reset_index()
    )

    # Computing the top x most used percent of keywords without the above high_rev/low_count exotics
    df = df[df.keyword_count >= 5]
    perc_thresh = 70  # chosen so that dataset is balanced
    perc = np.percentile(df.revenue_mean, perc_thresh)

    # Generating new column
    top_keywords = list(df[df.revenue_mean >= perc].Keywords)

    result_df = data.copy()[['id', 'Keywords']]
    result_df['has_top_keyword'] = result_df["Keywords"].apply(has_top_keyword, args=(top_keywords,))

    return result_df.drop(['Keywords'], axis=1), top_keywords


def getTimeFeatures(training_set):
    training_set = training_set.copy()
    releaseDate = pd.to_datetime(training_set['release_date']) 
    training_set["day"] = releaseDate.dt.dayofweek
    year = releaseDate.dt.year
    #some years are >2020 --> subtract 100
    year[year>2020] = year[year>2020]-100
    training_set["year"] = year
    training_set["age"] = year.max() - year
    return training_set[['id','day','year','age']]

def getNumericFeatures(training_set):
    training_set = training_set.copy()
    training_set["budgetLog"] = np.log1p(training_set['budget'])
    training_set["PopLog"] = np.log1p(training_set['popularity'])
    return training_set[['id','budgetLog','PopLog']]

def getBinaryFeatures(df):
    df = df.copy()
    df["hashomepage"] = ~(df["homepage"].isna())
    df["isinCollection"] = ~(df["belongs_to_collection"].isna())
    df["zeroBudget"] = (df["budget"]==0)
    return df[['id',"hashomepage",'isinCollection',"zeroBudget"]]

def getStarFeature(df):
    df = df.copy()
    df.loc[df.cast.isnull(), "cast"] = ''
    castList = df.cast.str.strip('[]')
    listOfallActors = pd.Series(pd.Series(list(", ".join(castList.unique().tolist()).split('}, '))).str.split("'name': '").str[1].str.split("'").str[0].tolist())
    allActors = listOfallActors.value_counts()
    topActors = allActors[allActors>=10].index
    df['hasStar'] = df.cast.apply(lambda row: 1 if any(act in row for act in topActors) else 0)
    df['NumStar']= df.cast.apply(lambda row: sum(act in row for act in topActors))
    return df[['id',"hasStar",'NumStar']]

#only works for genre at the moment, e.g., tranformListIntoBinaryFeatures(df, "genre", 100)
#also works for spoken_language and production_countries now
#also work for production_company
def tranformListIntoBinaryFeatures(df, feature, treshhold):
    df.list = df[feature].str.strip('[]')
    df.list[df.list.isnull()] = ''
    genres_list = pd.Series(list(set(", ".join(df.list.unique().tolist()).split('}, ')))).str.split("'name': '").str[1].str.split("'").str[0].tolist()
    
    for x in range(genres_list.count("")):
        genres_list.remove("")
    genres_list=['missing' if x is np.nan else x for x in genres_list]
    
    genres_list_trimmed = genres_list.copy()
    for genre in genres_list:
        df[genre] = df[feature].str.contains(genre)
        df[genre] = df[genre].fillna(False)
        if df[genre].sum(axis = 0) < treshhold :
            genres_list_trimmed.remove(genre)
        #else:
            #print(df[genre].sum(axis = 0))
            #print(genre)
    print(genres_list_trimmed)
    df["numberCount"] = df[genres_list].sum(axis = 1)
    return df[['id',"numberCount"]+genres_list_trimmed].copy()
