import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

# check language
def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res[0][0:2]
    return 'na'


# Read in data (Fill with na as missing may mean no views that day)
train = pd.read_csv('train_1.csv').fillna(0)
print(train.head())
print(train.info())

train['lang'] = train.Page.map(get_language)
print(Counter(train.lang))

