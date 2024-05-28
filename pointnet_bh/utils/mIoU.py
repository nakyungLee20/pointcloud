import os
import json
import numpy as np


def mIoU_per_class(gpath, ppath, num_class):
    with open(gpath, 'r') as fg:
        g_points = np.loadtxt(fg).astype(np.int64)
        
    with open(ppath, 'r') as fp:
        p_points = np.loadtxt(fp).astype(np.int64)
    
    ious = []
    for clss in range(num_class):
        gt_inds = (g_points == clss)
        pred_inds = (p_points == clss)
        
        intersect = np.sum(pred_inds & gt_inds)
        union = np.sum(pred_inds | gt_inds)
        
        if union == 0:
            ious.append(1)
        else:
            ious.append(intersect/union)
            
    return ious

def mean_mIoU(gpath, ppath, num_class):
    ious = mIoU_per_class(gpath, ppath, num_class)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    if not valid_ious:
        return np.nan
    return np.mean(valid_ious)


def pointnet(gpath, ppath, num_class):
    with open(gpath, 'r') as fg:
        g_points = np.loadtxt(fg)
        
    with open(ppath, 'r') as fp:
        p_points = np.loadtxt(fp)
    
    for shape_idx in range(g_points.shape[0]):
        parts = range(5)    #np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(p_points[shape_idx] == part, g_points[shape_idx] == part))
            U = np.sum(np.logical_or(p_points[shape_idx] == part, g_points[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
    
    return part_ious, np.mean(part_ious)


prediction = "/home/leena/pointnet/pointnet.pytorch/result/m_0524/"
groundtruth = "/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/0524/points_label/"
class_mious = np.empty((0,5))
mious = []
point_cl_mious = np.empty((0,5))
point_mious = []

for filename in os.listdir(prediction):
    if filename.endswith('.pts'):
        pred_path = prediction + filename
        gt_path = groundtruth + filename.rstrip(".pts")+".seg"
        
        # calculate with miou union = 1
        class_iou = mIoU_per_class(gt_path, pred_path, 5)
        class_iou = np.array(class_iou)
        class_mious = np.vstack((class_mious, class_iou))
        
        iou = mean_mIoU(gt_path, pred_path, 5)
        mious.append(iou)
        
        # calculate with miou union = 1
        #cl_ious, tot = pointnet(gt_path, pred_path, 5)
        #point_cl_mious = np.vstack((point_cl_mious, cl_ious))
        
        #point_mious.append(tot)


total_class = np.mean(class_mious, axis = 0)
total = np.mean(mious)
print("mIoU union = 1")
print("Total test dataset mIoU per class:", np.round(total_class, 4))
print("Total test dataset mIoU:", total)

# print()
# point_total_class = np.nanmean(point_cl_mious, axis = 0)
# point_total = np.mean(point_mious)
# print("mIoU union = 1")
# print("Total test dataset pointnet mIoU per class:", np.round(point_total_class, 4))
# print("Total test dataset pointnet mIoU:", point_total)

with open("class_mIoU_0524.json", "w") as file:
    json.dump(class_mious.tolist(), file)

# with open("class_mIoU_0524_pointnet.json", "w") as file:
#     json.dump(point_cl_mious.tolist(), file)
    