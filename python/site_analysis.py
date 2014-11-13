# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# <codecell>

from ozone_util import calculate_var,calculate_var2, calculate_entropy, ent_core, load_one_day_data, print_info
from impurity import calc_feature_impurity, calc_impurity, calc_mim, calc_weighted_impurity

# <codecell>

sample_path = "/home/kcai/data/kaggle_ctr/site3.csv"
raw_ctr = pd.read_csv(sample_path)
del raw_ctr['id']
raw_ctr['imps']=1
raw_ctr.shape

# <codecell>

# those features are all NaN
feature_to_go = ['app_id','app_domain','app_category']
for f in feature_to_go:
    assert(np.sum([ x == x  for x in raw_ctr[f]]) == 0 )
    del raw_ctr[f]

# <codecell>

# print np.sum([ x == x  for x in raw_ctr.app_id])
# print np.sum([ site == site  for site in raw_ctr.app_domain])
# ['site_domain', 'site_category', 'device_id', 'device_model']

for col in raw_ctr.columns:
    if  col !='imps':
        row_cnt =  np.sum([ site == site  for site in raw_ctr[col]])
        if row_cnt < raw_ctr.shape[0]:
            print col, row_cnt, raw_ctr[col].dtype, len(set(raw_ctr[col])) #, calc_feature_impurity(raw_ctr, col)
            # fill NaN with '---' so that we can handle it properly
            # the default should be 0 if it is numeric
            if raw_ctr[col].dtype == 'float64':
                raw_ctr[col][ [x != x for x in raw_ctr[col]]] = -99
            else:
                raw_ctr[col][ [x != x for x in raw_ctr[col]]] = '---'

# <codecell>

# truncate the data for convience now
oos = raw_ctr.tail(2000000)
raw_ctr = raw_ctr.head(7e6)
total_clicks = np.sum(raw_ctr.click)
print total_clicks
print np.sum(raw_ctr.imps)

# <codecell>

for col in raw_ctr.columns:
    if col !='imps':
        print col, len(set(raw_ctr[col])), calc_feature_impurity(raw_ctr, col)

# <codecell>

def agg_in_sample(raw_ctr):
    features = ['click','C1', u'banner_pos','site_category', 'device_os', u'device_make',
                'device_type', u'device_conn_type', u'device_geo_country',
                'C17', 'C18', 'C19', u'C20', u'C21', u'C22', u'C23', u'C24']

    agg_dict = {'imps':np.sum}
    raw_ctr = raw_ctr.groupby(features, as_index = False).agg(agg_dict)
    return raw_ctr

def make_dummy_feature_old(agged):
    feature_extract = ['C1', 'banner_pos', 'site_category', 'device_os', 'device_make', 'device_geo_country']
    feature_keep = ['imps', 'click', 'device_conn_type', 'C17','C18','C19','C20','C21','C22','C23','C24']
    count = True
    for feat in feature_extract:
        if count:
            output =  pd.get_dummies(agged[feat], prefix=feat)
            count = False
        else:
            output = output.join( pd.get_dummies(agged[feat], prefix=feat))
    
    return agged[feature_keep].join(output)

def my_join(df1, df2):
    return df1.join(df2)

def make_dummy_feature(agged):
    feature_extract = ['C1', 'banner_pos', 'site_category', 'device_os', 'device_make', 'device_geo_country']
    feature_keep = ['imps', 'click', 'device_conn_type', 'C17','C18','C19','C20','C21','C22','C23','C24']
    return reduce(my_join,[agged[feature_keep]] + [pd.get_dummies(agged[feat], prefix=feat) for feat in feature_extract])
        

# <codecell>

raw_ctr

# <codecell>

agged = agg_in_sample(raw_ctr)
in_sample = make_dummy_feature(agged)
print in_sample.shape
print np.sum(in_sample.imps)
print np.sum(in_sample.click)

# <codecell>

agged = agg_in_sample(oos)
oo_sample = make_dummy_feature(agged)
oo_sample.shape

# <codecell>

#plt.hist(raw_ctr.imps)
data = [math.log10(imp) for imp in agged.imps]
plt.hist(data, bins = 30, log= True)

# <codecell>

round(calc_weighted_impurity(raw_ctr, 'site_domain', 'site_category'),3)

# <codecell>

round(calc_weighted_impurity(raw_ctr, 'site_domain', 'site_id'),3)

# <codecell>

round(calc_weighted_impurity(raw_ctr, 'site_id', 'site_category'),3)

# <codecell>

round(calc_weighted_impurity(raw_ctr, 'C23', 'C18'),3)

# <codecell>

round(calc_weighted_impurity(raw_ctr, 'site_id', 'site_domain'),3)

# <markdowncell>

# # apply Random Forest

# <codecell>

from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier

# <codecell>

models = [DecisionTreeClassifier(max_depth=None),
          RandomForestClassifier(n_estimators=n_estimators),
          ExtraTreesClassifier(n_estimators=n_estimators),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             n_estimators=n_estimators)]

# <codecell>

model = RandomForestClassifier(n_estimators = 100, 
                               max_depth = None,
                               max_features = 'auto',
                               min_samples_split = 100,
                               min_samples_leaf = 30,
                               n_jobs = -1)

# <codecell>

# set up the in sample data 
y = np.array(in_sample.click)
X = np.array(in_sample[in_sample.columns[2:]])
sample_weight = np.array(in_sample.imps)
print X.shape
print  np.sum(sample_weight)

# <codecell>

np.sum(y * sample_weight)

# <codecell>

oos_y = np.array( oo_sample.click)
oos_X = np.array( oo_sample[oo_sample.columns[2:]])

# <codecell>

clf = model.fit(X, y, sample_weight = sample_weight) 

# <codecell>

clf.score(X, y, sample_weight) # 0.8

# <codecell>


# <codecell>

import scipy as sp

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def llfun_w(act, pred, weights):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred))
    return sum( ll *  weights) * -1.0/sum(weights)

# <codecell>

print llfun  ([1,1, 1, 0], [1,0.5, 0.5, 0.3])
print llfun_w([1,1,0], [1,0.5,  0.3], [1,2,1])

# <codecell>

y_hat = clf.predict_proba(X)
llfun_w(y, [ tmp[1] for tmp in y_hat], sample_weight)  # 0.451

# <codecell>

oos_y_hat = clf.predict_proba(oos_X)
llfun(oos_y, [ tmp[1] for tmp in oos_y_hat])  # 0.44

# <codecell>

