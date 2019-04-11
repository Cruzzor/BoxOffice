#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import ast

# Function for extracting names out of a keywords list 
# Also replaces spaces by underscores for better further processing
def extract_names(keyword_list):
    if pd.isna(keyword_list):
        return []
    else:
        return list(map(lambda x: x['name'].replace(' ', '_'), ast.literal_eval(keyword_list)))


# Compute sum and mean revenue for each keyword + count occurences
def f(x):
    d = {}
    d['revenue_sum'] = x['revenue'].sum()
    d['revenue_mean'] = x['revenue'].mean()
    d['keyword_count'] = len(x['revenue'])
    return pd.Series(d, index=['revenue_sum', 'revenue_mean', 'keyword_count'])


def has_top_keyword(keyword_list):
    if not pd.isna(keyword_list):
        list_of_keywords = extract_names(keyword_list)
        for keyword in list_of_keywords:
            if keyword in top_keywords:
                return True
    return False