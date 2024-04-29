#%%
import pandas as pd

import numpy as np

df = pd.read_csv('../../Datasets/DNN-EdgeIIoT-dataset.csv', low_memory=False) 

df.head(5)

print(df['Attack_type'].value_counts())

from sklearn.utils import shuffle

drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4", 

         "http.file_data","http.request.full_uri","icmp.transmit_timestamp",

         "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",

         "tcp.dstport", "udp.port", "mqtt.msg"]

df.drop(drop_columns, axis=1, inplace=True)

df.dropna(axis=0, how='any', inplace=True)

df.drop_duplicates(subset=None, keep="first", inplace=True)

df = shuffle(df)

df.isna().sum()

print(df['Attack_type'].value_counts())

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

def encode_text_dummy(df, name):

    dummies = pd.get_dummies(df[name])

    for x in dummies.columns:

        dummy_name = f"{name}-{x}"

        df[dummy_name] = dummies[x]

    df.drop(name, axis=1, inplace=True)

encode_text_dummy(df,'http.request.method')

encode_text_dummy(df,'http.referer')

encode_text_dummy(df,"http.request.version")

encode_text_dummy(df,"dns.qry.name.len")

encode_text_dummy(df,"mqtt.conack.flags")

encode_text_dummy(df,"mqtt.protoname")

encode_text_dummy(df,"mqtt.topic")


df.info()
df['Attack_type'] = df['Attack_type'].astype('category')
df["Attack_label"] = df["Attack_label"].astype('category')
#%%

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#Get correlation of all the features of the dataset
df_corr = df.copy()
df_corr['Attack_label'] = df_corr['Attack_type'].cat.codes
df_corr['Attack_type'] = df_corr['Attack_type'].cat.codes
corr_matrix = df_corr.corr()

#Get the features that are highly correlated with the target
print(corr_matrix["Attack_label"].abs().sort_values(ascending=False))

print(corr_matrix["Attack_type"].abs().sort_values(ascending=False))


#Get the labels of most correlated features in matrix
corr_pairs = corr_matrix.unstack()
top_correlated_pairs = corr_pairs.abs().sort_values(ascending=False)
top_correlated_pairs = top_correlated_pairs[
    (top_correlated_pairs.index.get_level_values(0) != top_correlated_pairs.index.get_level_values(1))
]
# Filter to keep only unique pairs (ignore mirror entries and self-correlation)
top_correlated_pairs = top_correlated_pairs[
    (top_correlated_pairs.index.get_level_values(0) != top_correlated_pairs.index.get_level_values(1))
]
print(top_correlated_pairs)

#Plot the correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix, annot=True, fmt=".2f")
plt.show()


#%%
#get X and Y
X = df.drop(['Attack_type', 'Attack_label'], axis=1)

Y_multi = df['Attack_type'].cat.codes
Y_binary = df['Attack_label']


#%%

# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def process_x_y(X, Y, label):
    save_path = '../../Datasets/Edge-IIoT-pre-processed/'
    X = X.copy()
    Y = Y.copy()
    X = X.astype(np.float32)   
    Y = Y.astype(np.float32)
    X = X.to_numpy()
    Y = Y.to_numpy()
    #standardize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    #split the data
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    #Print size before SMOTE
    # print("Before SMOTE")
    # print(np.unique(Y_train, return_counts=True))
    # print(np.unique(Y_test, return_counts=True))


    # #SMOTE
    # smote = SMOTE(sampling_strategy='auto', random_state=42)
    # X_train, Y_train = smote.fit_resample(X_train,Y_train)

    #Print size after SMOTE
    # print("After SMOTE")
    # print(np.unique(Y_train, return_counts=True))
    # print(np.unique(Y_test, return_counts=True))

    #get validation set
    # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    #Shape before one hot encoding
    print("Before one hot encoding")
    print(X.shape)
    print(Y.shape)

    #One hot encoding
    enc = OneHotEncoder()
    Y = Y.reshape(-1,1)
    Y = enc.fit_transform(Y).toarray()

    #Shape after one hot encoding
    print("After one hot encoding")
    print(X.shape)
    print(Y.shape)

    #save the data
    np.save(save_path + 'X.npy', X)
    np.save(save_path + 'Y.npy', Y)

process_x_y(X, Y_multi, 'Multi')
# process_x_y(X, Y_binary, 'Binary')

# %%
