# SHM-Semi

1. Generate one-channel time-history gray scale images and save them under './dataset/time_history_01_120_100/'. In this study, the image resolution used is   120 x 100 and only time-history information is used. The txt file "201201.txt" saves the image names and ground truth labels. 

2. Run Python test_model.py to test the model performance on test data. The model weight used is under "./anomaly@1400-timehistory-semisupervised/", which is trained using 1400 labelled data and available unlabelled data. The results will be saved under "./result/."

   For test_model.py, the data path information in line 53-58 should be changed. Also, in line 256, the data path information in the function "get_anomaly" should be checked. 
   
   
