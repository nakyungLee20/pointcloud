import numpy as np
import os

def convert_label(seg_path):
    with open(seg_path, "r") as file:
        labels = np.loadtxt(file) 
    
    fname = seg_path.split('0603/points_label_ori/')[1]
    save_path = '/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/0603/points_label/'+fname
    
    with open(save_path, "w") as file2:
        for label in labels:
            label = int(label)
            if label == 0:
                label = 50
            elif label == 1:
                label = 51
            elif label == 2:
                label = 52
            
            file2.write(f"{label}\n")
            
    print(f"Converting {fname} done.")


root = "/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/0603/points_label_ori"
for filename in os.listdir(root):
    if filename.endswith('.seg'):
        seg_path = os.path.join(root, filename)
        convert_label(seg_path)
