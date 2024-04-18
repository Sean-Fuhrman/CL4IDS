#%%
import pandas as pd

import numpy as np

df = pd.read_csv('../../Datasets/X-IIoTID dataset.csv', low_memory=False) 


df = df.drop(df.columns[df.nunique() == 1], axis=1)


drop_cols = ['Date', 'Timestamp', 'Scr_IP', 'Scr_port', 'Des_IP', 'Des_port',
       'Protocol']
df = df.drop(drop_cols, axis=1)


#Drop Nan
df = df.dropna()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#Encode objects as ints
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

float_cols = ['Duration', 'Scr_bytes', 'Des_bytes', 'missed_bytes', 'Scr_pkts',
              'Scr_ip_bytes', 'Des_pkts', 'Des_ip_bytes', 'total_bytes', 'total_packet',
              'paket_rate', 'byte_rate', 'Scr_packts_ratio', 'Des_pkts_ratio',
              'Scr_bytes_ratio', 'Des_bytes_ratio', 'Avg_ideal_time', 'Avg_user_time',
              'Avg_nice_time', 'Avg_system_time', 'Avg_iowait_time', 'Avg_tps',
              'Avg_rtps', 'Avg_wtps', 'Avg_ldavg_1', 'Std_ideal_time', 'Std_user_time',
              'Std_nice_time', 'Std_system_time', 'Std_iowait_time', 'Std_tps',
              'Std_rtps', 'Std_wtps', 'Std_ldavg_1', 'Avg_kbmemused', 'Std_kbmemused',
              'Avg_num_Proc/s', 'Std_num_proc/s', 'Avg_num_cswch/s', 'std_num_cswch/s'
              ]

for col in float_cols:
    df[col] = df[col].replace('-', '0')
    df[col] = df[col].replace('?', '0')
    df[col] = df[col].replace(' ', '0')
    df[col] = df[col].replace('aza', '0')
    df[col] = df[col].replace('excel', '0')
    df[col] = df[col].replace('#DIV/0!', '0')
    df[col] = df[col].astype(float)

df["anomaly_alert"] = df["anomaly_alert"].replace('-', 'FALSE')




for col in df.columns:
    if df[col].dtype == 'object':
        #Attempt to convert to float
        try:
            df[col] = df[col].astype(float)
        except:
            #Use categorical label encoder
            df[col] = le.fit_transform(df[col])

#Convert to float
df = df.astype(float)

print(df["class1"].value_counts())

#%%

X = df.drop(["class1", "class2", "class3"], axis=1)

Y_1 = df['class1']
Y_2 = df['class2']
Y_3 = df['class3']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

X = X.to_numpy()
Y_1 = Y_1.to_numpy()
Y_2 = Y_2.to_numpy()
Y_3 = Y_3.to_numpy()

#Print shapes
print(X.shape)
print(Y_1.shape)
print(Y_2.shape)
print(Y_3.shape)

#One hot encoding
enc = OneHotEncoder()
# Y_1 = enc.fit_transform(Y_1.reshape(-1, 1)).toarray()
# Y_2 = enc.fit_transform(Y_2.reshape(-1, 1)).toarray()
# Y_3 = enc.fit_transform(Y_3.reshape(-1, 1)).toarray()

#Standardize X
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Print shapes
print(X.shape)
print(Y_1.shape)
print(Y_2.shape)
print(Y_3.shape)

print(np.unique(Y_1, return_counts=True))

X_train, X_test, Y_1_train, Y_1_test, Y_2_train, Y_2_test, Y_3_train, Y_3_test = train_test_split(X, Y_1, Y_2, Y_3, test_size=0.2, random_state=42)


# %%
#get validation set
X_train, X_val, Y_1_train, Y_1_val, Y_2_train, Y_2_val, Y_3_train, Y_3_val = train_test_split(X_train, Y_1_train, Y_2_train, Y_3_train, test_size=0.2, random_state=42)

save_path = '../../Datasets/X-IIoT-pre-processed/Custom PreProcessed/'

np.save(save_path + 'X_train.npy', X_train)
np.save(save_path + 'X_val.npy', X_val)
np.save(save_path + 'X_test.npy', X_test)
np.save(save_path + 'Y_1_train.npy', Y_1_train)
np.save(save_path + 'Y_1_val.npy', Y_1_val)
np.save(save_path + 'Y_1_test.npy', Y_1_test)
np.save(save_path + 'Y_2_train.npy', Y_2_train)
np.save(save_path + 'Y_2_val.npy', Y_2_val)
np.save(save_path + 'Y_2_test.npy', Y_2_test)
np.save(save_path + 'Y_3_train.npy', Y_3_train)
np.save(save_path + 'Y_3_val.npy', Y_3_val)
np.save(save_path + 'Y_3_test.npy', Y_3_test)


# %%
