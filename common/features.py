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