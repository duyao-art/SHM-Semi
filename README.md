# SHM-Semi

1. Generate one-channel time-history gray scale image and save them under './dataset/time_history_01_120_100/'. In this study, the image resolution used is   120 x 100 and only time-history information is used. The txt file "201201.txt" saves the image name and ground truth label. 

2. Run Python test_model.py to test the model performance on test data. 
   the data path information in line 53-58 should be changed. Also, in line 256, the data path information in the function "get_anomaly" should be checked.
   
   
