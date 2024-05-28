import json
import os
import random

source = "/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/0216/points"
new_source = []
for filename in os.listdir(source):
    if filename.endswith('.pts'):
        ele = "shape_data/0216/"+filename.strip(".pts")
        new_source.append(ele)

#print(new_source)
total = len(new_source)

# test file
with open('train_test_split/test_file_list.json', 'w') as f:
    json.dump(new_source, f)

