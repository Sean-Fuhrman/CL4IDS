# EDA Notes

820,834 total rows
Dropping Na leaves 820,502 

Bad_checksum and is_SYN_with_RST are always false for this dataset.

Some columns that are numerical contain "-".

QUESTION: is it ok to convert all "-" into 0? A column may already contain 0's and "-"s.

Input size = 56
Num Classes = 19

class1_num_to_label = {0: 'BruteForce', 1: 'C&C', 2: 'Dictionary', 3: 'Discovering_resources', 4: 'Exfiltration', 5: 'Fake_notification', 6: 'False_data_injection', 7: 'Generic_scanning', 8: 'MQTT_cloud_broker_subscription', 9: 'MitM', 10: 'Modbus_register_reading', 11: 'Normal', 12: 'RDOS', 13: 'Reverse_shell', 14: 'Scanning_vulnerability', 15: 'TCP Relay', 16: 'crypto-ransomware', 17: 'fuzzing', 18: 'insider_malcious'}

## Class distribution 

Class 1 - 19 classes (Same as preproccessed folder)

Attack Values

- Normal                            421417
- RDOS                              141261
- Scanning_vulnerability             52852
- Generic_scanning                   50277
- BruteForce                         47241
- MQTT_cloud_broker_subscription     23524
- Discovering_resources              23148
- Exfiltration                       22134
- insider_malcious                   17447
- Modbus_register_reading             5953
- False_data_injection                5094
- C&C                                 2863
- Dictionary                          2572
- TCP Relay                           2119
- fuzzing                             1313
- Reverse_shell                       1016
- crypto-ransomware                    458
- MitM                                 117
- Fake_notification                     28

Note: All Fake_notification have NaN dates. 

Class 2 - 10 classes

Attack Values

- Normal               421417
- RDOS                 141261
- Reconnaissance       127590
- Weaponization         67260
- Lateral _movement     31596
- Exfiltration          22134
- Tampering              5122
- C&C                    2863
- Exploitation           1133
- crypto-ransomware       458

Class 3 - 2  classes (Normal v. Attack)

Attack Values 

- Normal    421417
- Attack    399417
## Correlation Information

Columns with >0.1 correlation for class 1:
- Avg_system_time                0.309402
- Avg_nice_time                  0.305173
- OSSEC_alert_level              0.298863
- OSSEC_alert                    0.292608
- class3                         0.252690
- Avg_num_cswch/s                0.208780
- FIN or RST                     0.208179
- Avg_kbmemused                  0.205753
- is_syn_only                    0.204745
- read_write_physical.process    0.199484
- class2                         0.197534
- Is_SYN_ACK                     0.164808
- is_pure_ack                    0.164266
- Scr_packts_ratio               0.127340
- anomaly_alert                  0.121094
- Std_system_time                0.114903
- Des_bytes                      0.102751
- Avg_ldavg_1                    0.101586

Columns with >0.1 correlation for class3:
- class2                         0.501287
- Avg_ideal_time                 0.397708
- Scr_bytes                      0.394129
- Avg_ldavg_1                    0.379435
- is_with_payload                0.339302
- Std_system_time                0.291980
- Conn_state                     0.288874
- Des_pkts_ratio                 0.258388
- Des_bytes_ratio                0.255001
- class1                         0.252690
- Avg_num_cswch/s                0.235075
- paket_rate                     0.228018
- read_write_physical.process    0.215794
- OSSEC_alert                    0.215339
- OSSEC_alert_level              0.213853
- Des_ip_bytes                   0.189726
- Avg_kbmemused                  0.172943
- Des_pkts                       0.167881
- total_packet                   0.165854
- total_bytes                    0.161108
- FIN or RST                     0.141449
- Scr_pkts                       0.133306
- Avg_system_time                0.131118
- is_syn_only                    0.129644
- Duration                       0.118397
- Avg_nice_time                  0.117487
- Des_bytes                      0.103696
- std_num_cswch/s                0.103346

Columns with high mutual correlation:

Process_activity  Succesful_login     0.999250
Succesful_login   Process_activity    0.999250
                  is_privileged       0.998169
is_privileged     Succesful_login     0.998169
Process_activity  is_privileged       0.997447
is_privileged     Process_activity    0.997447
Is_SYN_ACK        is_pure_ack         0.996282
is_pure_ack       Is_SYN_ACK          0.996282
is_syn_only       FIN or RST          0.982299
FIN or RST        is_syn_only         0.982299
dtype: float64
Process_activity   Succesful_login      0.999250
Succesful_login    Process_activity     0.999250
                   is_privileged        0.998169
is_privileged      Succesful_login      0.998169
Process_activity   is_privileged        0.997447
is_privileged      Process_activity     0.997447
Is_SYN_ACK         is_pure_ack          0.996282
is_pure_ack        Is_SYN_ACK           0.996282
is_syn_only        FIN or RST           0.982299
FIN or RST         is_syn_only          0.982299
OSSEC_alert        OSSEC_alert_level    0.978445
OSSEC_alert_level  OSSEC_alert          0.978445
Succesful_login    Login_attempt        0.970894
Login_attempt      Succesful_login      0.970894
Process_activity   Login_attempt        0.970166
Login_attempt      Process_activity     0.970166
is_privileged      Login_attempt        0.969116
Login_attempt      is_privileged        0.969116
File_activity      is_privileged        0.933577
is_privileged      File_activity        0.933577
File_activity      Succesful_login      0.931882
Succesful_login    File_activity        0.931882
File_activity      Process_activity     0.931025
Process_activity   File_activity        0.931025
Login_attempt      File_activity        0.904759
File_activity      Login_attempt        0.904759
is_with_payload    Conn_state           0.841258
Conn_state         is_with_payload      0.841258
Avg_num_Proc/s     Std_num_proc/s       0.823956
Std_num_proc/s     Avg_num_Proc/s       0.823956

## Protocol Breakdown

### TCP

Tcp contains 422,002 (about half) of the rows, and 16 of the attacks (class 1).
No columns are 0 when selecting for TCP protocol

### UDP

UDP contains 395,620 (still about half) of the rows, and 6 of the attacks (class 1)
8 columns are 1 unique value when selecting UDP protocol


### ICMP

ICMP contains 2,726 of the rows, and 7 of the attacks (class 1)
11 columns are 1 unique value when selecting for ICMP

### ?

The fourth protocol is "?" which contains 154 rows that are all crpyto-ransomware attacks

The column "Anomoly Alert" contains True, False, values. However for these 154 rows it contains "-". I am defaulting this to False