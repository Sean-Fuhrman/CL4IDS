# EDA Notes

## Dataset Notes

The following columns only contain 1 unique value: 
- icmp.unused            1
- http.tls_port          1
- dns.qry.type           1
- mqtt.msg_decoded_as    1

The following columns only contain 2 unique values:
- arp.hw.size                  2
- http.response                2
- tcp.connection.fin           2
- tcp.connection.rst           2
- tcp.connection.syn           2
- tcp.connection.synack        2
- tcp.flags.ack                2
- dns.retransmit_request       2
- dns.retransmit_request_in    2
- mqtt.conflag.cleansess       2
- mqtt.conflags                2
- mqtt.proto_len               2
- mqtt.topic_len               2
- mqtt.ver                     2
- Attack_label                 2

There is a difference between '0' and '0.0' with how the dataset is created. Filtering by rows with '0.0' in certain columns leaves mostly attacks remaining. 


Attack_num_to_label = {0: 'Backdoor', 1: 'DDoS_HTTP', 2: 'DDoS_ICMP', 3: 'DDoS_TCP', 4: 'DDoS_UDP', 5: 'Fingerprinting', 6: 'MITM', 7: 'Normal', 8: 'Password', 9: 'Port_Scanning', 10: 'Ransomware', 11: 'SQL_injection', 12: 'Uploading', 13: 'Vulnerability_scanner', 14: 'XSS'}

## Class Distribution
- 72.802914% Normal
- 27.187086% Attack

More Specifically:
- Normal                   72.802914
- DDoS_UDP                  5.478008
- DDoS_ICMP                 5.246753
- SQL_injection             2.307272
- Password                  2.259958
- Vulnerability_scanner     2.258020
- DDoS_TCP                  2.255857
- DDoS_HTTP                 2.249053
- Uploading                 1.695836
- Backdoor                  1.120313
- Port_Scanning             1.016762
- XSS                       0.717150
- Ransomware                0.492294
- MITM                      0.054704
- Fingerprinting            0.045106

Values: 
- Normal                   1615643
- DDoS_UDP                  121568
- DDoS_ICMP                 116436
- SQL_injection              51203
- Password                   50153
- Vulnerability_scanner      50110
- DDoS_TCP                   50062
- DDoS_HTTP                  49911
- Uploading                  37634
- Backdoor                   24862
- Port_Scanning              22564
- XSS                        15915
- Ransomware                 10925
- MITM                        1214
- Fingerprinting              1001

## Correlation Matrix

A lot of highly correlated features, particullarly with mqtt protocol. Long list of features can be seen running default_edge_preprocess.py. Many have 1.0 correlation

Most correlated features with Attack_label (Absolute value of correlation > 0.1):
- tcp.flags.ack                           0.401246
- http.request.method-0.0                 0.396402
- http.request.version-0.0                0.395986
- tcp.flags                               0.390703
- icmp.seq_le                             0.336649
- icmp.checksum                           0.329774
- http.request.version-HTTP/1.1           0.305571
- http.request.method-0                   0.300535
- http.request.version-0                  0.300134
- http.request.method-GET                 0.296032
- http.response                           0.284858
- tcp.ack_raw                             0.248732
- udp.stream                              0.246157
- tcp.checksum                            0.231791
- mqtt.topic-0.0                          0.157384
- mqtt.conack.flags-0.0                   0.157384
- mqtt.protoname-0.0                      0.157384
- dns.qry.name.len-0.0                    0.157320
- dns.qry.name.len-0                      0.157015
- mqtt.topic-0                            0.142094
- mqtt.protoname-0                        0.142093
- mqtt.conack.flags-0                     0.142087
- http.referer-0.0                        0.130971
- http.referer-0                          0.127645
- tcp.ack                                 0.113267
- http.content_length                     0.100150

Most correlated features with Attack_type (Absolute value of correlation > 0.1):
- tcp.flags.ack                           0.401246
- http.request.method-0.0                 0.396402
- http.request.version-0.0                0.395986
- tcp.flags                               0.390703
- icmp.seq_le                             0.336649
- icmp.checksum                           0.329774
- http.request.version-HTTP/1.1           0.305571
- http.request.method-0                   0.300535
- http.request.version-0                  0.300134
- http.request.method-GET                 0.296032
- http.response                           0.284858
- tcp.ack_raw                             0.248732
- udp.stream                              0.246157
- tcp.checksum                            0.231791
- mqtt.topic-0.0                          0.157384
- mqtt.conack.flags-0.0                   0.157384
- mqtt.protoname-0.0                      0.157384
- dns.qry.name.len-0.0                    0.157320
- dns.qry.name.len-0                      0.157015
- mqtt.topic-0                            0.142094
- mqtt.protoname-0                        0.142093
- mqtt.conack.flags-0                     0.142087
- http.referer-0.0                        0.130971
- http.referer-0                          0.127645
- tcp.ack                                 0.113267
- http.content_length                     0.100150

## Protocol Breakdown

This dataset contains data from different protocols, some which are exclusive from others.

### IP

The ip.src_hot / ip.dst_host are from the ips of the devices used to create the dataset. Certain IP address are used for certain attacks / normal settings. For example 192.168.0.101 is only used for MITM attacks and normal. This is reason to drop IP columns. 

### TCP

To analyze which rows are TCP rows, I select rows that contain non-zero tcp.srcports. If the tcp.srcport is 0, then the tcp.dstport is also 0, so I think this means I can assume if tcp.srcport is 0 that row does not contain TCP information.
1,893,445

There are 146,526 rows with no TCP src port
of which :
-  5,854 are ARP
-  9,873 are all zeros besides ['frame.time', 'Attack_type', 'Attack_label','ip.src_host', 'ip.dst_host']
-  2,227 are UDP port (All Normal)
-  128,572 other

Given this, most of the dataset has TCP src ports
Of the 1,893,812 rows:
- 0 contain ARP data
- 367 contain UDP data and have 0 tcp.dst port, all of which are MITM attacks. These are the only MITM attacks in this section of dataset
- 0 contain ICMP data
- 227 contain DNS data
- These leaves HTTP, mqtt, mbtcp, as some protocols remaining.

### ARP

There are only 5,854 valid ARP rows (arp.opcode != 0)

If arp.opcode == 0.0 then hw.size == 0.0
Else hw.size == 6.0

For all these ARP rows, no other protocol is used, so ALL other values are all 0. Except for Attack labels / types.

There are 13 attack types used with ARP, however I do not see how any system could distinguish any ARP Attack vs. Normal

Distribution of ARP attacks:

Normal                   3071
Port_Scanning            1746
Ransomware                505
Backdoor                  324
Fingerprinting             82
XSS                        59
Password                   19
SQL_injection              15
Vulnerability_scanner      13
Uploading                   8
DDoS_ICMP                   7
DDoS_HTTP                   4
DDoS_TCP                    1

QUESTIONS:
Should this be dropped? What does it mean for an ARP network protocol to be "Randsom" or "Port scanning" attack? 

If it's just used to resolve MAC addresses. 

### UDP

There are only 2,594 valid UDP rows (udp.port != 0)

Attack Type distribution:
- 2,227 Normal
- 367 Man-in-the-middle 
In the paper it says that they simulate UDP DDOS flooding attacks, so where are they? Would they have udp.port = 0? 

If udp.port != 0, and udp.stream == 0.0, then it is a Man-in-the-middle attack

if udp.port != 0, and tcp.srcport != 0, then it is a man in the middle attack 

lets say udp.port is dropped:   
    If udp.stream != 0.0 then it is Normal
    If udp.stream == 0 and udp.time_delta != 0 then it is MITM


### HTTP

I think in this dataset, because it is using edge device??. Any http request is an attack. Could be reason to drop.

Selected http protocol by excluduing '0.0' http.request.method leaves only attack data. 

http.tls_port is always 0. 


Contains 209636
Attack type
SQL_injection            51203
Password                 50153
DDoS_HTTP                49911
Uploading                37634
Vulnerability_scanner    20735
(All of these attacks)

If exclud all zero http.request.method ('0.0' and '0'):

Contains 32,087 rows. Still only attack data

Attack_type
Vulnerability_scanner    21117
SQL_injection             4379
Password                  3306
Uploading                 1907
XSS                       1376
DDoS_HTTP                    2

if http.response = 1, then is it never normal:

19051 Rows
Attack_type
Vulnerability_scanner    8672
SQL_injection            4353
Password                 3420
Uploading                1904
DDoS_HTTP                 702
http.file_data contains literal http file strings. Like <!DOCTYP....

If any column that begins with http.* is non-zero '0' or '0.0', then it is an attack.

### ICMP

If icmp.checksum != 0: it leaves the following remaining
Attack_type
DDoS_ICMP         116425
Fingerprinting       581
Normal                 2
Name: count, dtype: int64

If icmp.seq_le != 0, it is not Normal:
Attack_type
DDoS_ICMP         116428
DDoS_UDP           11533
Fingerprinting       166
Name: count, dtype: int64

if icmp.transmit_timestamp != 0: it is fingerprinting

Attack_type
Fingerprinting    83

### DNS
If dns.qry.name.len is not 0, this leaves on 2,072 rows. 1,845 of which are normal, 227 are MITM

All 227 of these MITM overlap with the 367 MITM from UDP.

All rows with non-zero dns.qry.name.len have none-zero udp.timedelta. In other words all DNS stuff in this dataset is a subset of UDP stuff.Which I think should be dropped
### MQTT

MQTT seems to just be used by normal activity. Any non zero mqtt row I investigated only contains normal data. 

For instance: 
If mqtt.msgtype is non-zero, it is normal.
If mqtt.msgtype is zero. all other mqtt rows are zero except for 
mqtt.conack.flags
If mqtt.conack.flags is non-zero, it is normal.
Therefore, any row with MQTT data is always normal. 


### MBTCP
only 3 columns are mbtcp. mbtcp.len, mbtcp.trans_id, & mbtcp.unit_id.

If any of the 3 columns above are not 0:

150 rows remain, it is normal


## Recommended Preprocessing.

I feel like all protocols have their quirks, besides TCP. in other words, the other columns provide obvious ways to detect if an attack is malicous or not. 
Fore example any MQTT column indicates normal. Any http column indicates attack.
