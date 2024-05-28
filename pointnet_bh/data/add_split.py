import json
import os
import random

source = "/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/0524/points"
new_source = []
for filename in os.listdir(source):
    if filename.endswith('.pts'):
        ele = "shape_data/0524/"+filename.strip(".pts")
        new_source.append(ele)

#print(new_source)
total = len(new_source)
tr_src = new_source[:int(total*0.7)]
tt_src = new_source[int(total*0.7):]

# Fix original files
tr_file = "/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/train_test_split/shuffled_train_file_list_ori.json"
tt_file = "/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/train_test_split/shuffled_test_file_list_ori.json"

# train file
with open(tr_file, "r") as file:
    train_list = json.load(file)

print("Original train list", len(train_list))
train_list.extend(tr_src)
random.shuffle(train_list)
print("Extended train list", len(train_list))

with open('train_test_split/shuffled_train_file_list.json', 'w') as f:
    json.dump(train_list, f)

# test file
with open(tt_file, 'r') as file2:
    test_list = json.load(file2)

print("Original test list", len(test_list))
test_list.extend(tt_src)
random.shuffle(test_list)
print("Extended test list", len(test_list))

with open('train_test_split/shuffled_test_file_list.json', 'w') as f2:
    json.dump(test_list, f2)


with open('train_test_split/test_list.json', 'w') as f3:
    json.dump(tt_src, f3)
