#%%
import pandas as pd

import numpy as np

df = pd.read_csv('../Datasets/DNN-EdgeIIoT-dataset.csv', low_memory=False) 

df.head(5)
print(df.columns)

print(df['Attack_type'].value_counts())
#%%
from sklearn.utils import shuffle

# drop_columns = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.dst.proto_ipv4',
#        'arp.opcode', 'arp.hw.size', 'arp.src.proto_ipv4', 'icmp.checksum',
#        'icmp.seq_le', 'icmp.transmit_timestamp', 'icmp.unused',
#        'http.file_data', 'http.content_length', 'http.request.uri.query',
#        'http.request.method', 'http.referer', 'http.request.full_uri',
#        'http.request.version', 'http.response', 'http.tls_port', 
#        'tcp.dstport','tcp.options', 'tcp.payload', 'tcp.srcport', 'udp.port', 'udp.stream',
#        'udp.time_delta', 'dns.qry.name', 'dns.qry.name.len', 'dns.qry.qu',
#        'dns.qry.type', 'dns.retransmission', 'dns.retransmit_request',
#        'dns.retransmit_request_in', 'mqtt.conack.flags',
#        'mqtt.conflag.cleansess', 'mqtt.conflags', 'mqtt.hdrflags', 'mqtt.len',
#        'mqtt.msg_decoded_as', 'mqtt.msg', 'mqtt.msgtype', 'mqtt.proto_len',
#        'mqtt.protoname', 'mqtt.topic', 'mqtt.topic_len', 'mqtt.ver',
#        'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id']


df.drop(drop_columns, axis=1, inplace=True)

df.dropna(axis=0, how='any', inplace=True)

df.drop_duplicates(subset=None, keep="first", inplace=True)

df = shuffle(df)

df.isna().sum()

print(df['Attack_type'].value_counts())
#%%
df.info()
#%%
# drop all rows with all 0 values except for Attack_type and Attack_label
df = df.loc[~(df.drop(['Attack_type', 'Attack_label'], axis=1) == 0).all(axis=1)]
#%%
# Convert attack type to numerical values
df['Attack_type'] = df['Attack_type'].astype('category')
df.head()
#%%

X = df.drop(['Attack_type', 'Attack_label'], axis=1)
Y = df['Attack_type'].cat.codes
#%%
#get number of attack types
num_classes = len(df['Attack_type'].unique())
df['Attack_type'].value_counts()
#%%
print(num_classes)
#%%
#convert to numpy array
X = X.to_numpy()
Y = Y.to_numpy()
print(X.dtype)
print(X.shape)
print(Y.shape)

#%%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
#%%
save_path = '../Datasets/Edge-IIoT-pre-processed/'
np.save(save_path + 'X_train.npy', X_train)
np.save(save_path + 'X_test.npy', X_test)
np.save(save_path + 'Y_train.npy', Y_train)
np.save(save_path + 'Y_test.npy', Y_test)

#%%
#%%
df.to_csv('../Datasets/Edge-IIoT-preprocessed_DNN.csv', encoding='utf-8')
# %%


