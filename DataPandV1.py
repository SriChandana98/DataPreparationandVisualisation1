import numpy as np
import pandas as pd

data=pd.read_csv(r"scrapped_data_large.csv")
data.nunique()

data

data.shape

data.info()

#1 List of players who played for India
In=data.loc[data['teams']=='India,']
In

In.nunique()

In.shape[0] # Number of players who played for India

#2
team_vals = data['teams'].unique().tolist()
team_vals.remove('India,')

data['teams']=data['teams'].replace(to_replace=r'^India,', value='1', regex=True)
data['teams']=data['teams'].replace(team_vals,'0')

data['teams']

features=data.iloc[:,4:]
features

features = features.astype('str')
features.dtypes

for c in features.columns:
    features[c]=features[c].map(lambda x:x.rstrip('*,+,?,!'))
features=features.replace('-','0')

features

cols = features.columns
cols

features[cols] = features[cols].apply(pd.to_numeric, errors='coerce')
features.dtypes

features.fillna(np.nan)

X=np.array(features)
X

Y=np.array(data['teams']).T
Y

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X, Y)

importance_scores = rfc.feature_importances_
importance_scores

feature_importances = [(feature, score) for feature, score in zip(features.columns, importance_scores)]
feature_importances

feature_importances_sorted=sorted(feature_importances, key=lambda x: -x[1])[:3]
feature_importances_sorted ##dominant features or features responsible for selection

result1=pd.DataFrame(feature_importances_sorted,columns = ['features_not_imputed', 'score_not_imputed'])

result1

#3 Using KNNImputer
data=pd.read_csv(r"scrapped_data_large.csv")

team_vals = data['teams'].unique().tolist()
team_vals.remove('India,')

data['teams']=data['teams'].replace(to_replace=r'^India,', value='1', regex=True)
data['teams']=data['teams'].replace(team_vals,'0')
data['teams']

features=data.iloc[:,4:]
features

features = features.astype('str')
features.dtypes

for c in features.columns:
    features[c]=features[c].map(lambda x:x.rstrip('*,+,?,!'))

cols = features.columns
cols

features[cols] = features[cols].apply(pd.to_numeric, errors='coerce')
features.dtypes

from missingpy import KNNImputer

imputer = KNNImputer()
X_imputed = imputer.fit_transform(features)
X_imputed

Y=np.array(data['teams']).T
Y

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_imputed, Y)

importance_scores_imputed = rfc.feature_importances_
importance_scores_imputed

feature_importances_imputed = [(feature, score) for feature, score in zip(features.columns, importance_scores_imputed)]
feature_importances_imputed

feature_importances_sorted_imputed=sorted(feature_importances_imputed, key=lambda x: -x[1])[:3]
feature_importances_sorted_imputed

result2=pd.DataFrame(feature_importances_sorted_imputed,columns = ['features_imputed', 'score_imputed'])
result2

#comparing the values when imputed and not imputed.
result=result1.join(result2)
result 
#yes the result changes when KNNImputer is used
