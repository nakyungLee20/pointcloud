import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=5000,
                 classification=False,
                 class_choice=None,
                 split='test',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)
        self.npoints = point_set.shape[0]

        # choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        #point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        #seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        path = fn[1].split('0524/points/')[1].rstrip('.pts')

        if self.classification:
            return point_set, cls, path
        else:
            return point_set, seg, path

    def __len__(self):
        return len(self.datapath)

test_dataset = ShapeNetDataset(
    root='/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0',
    classification=False,
    class_choice=['Bokha'],
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4)
print("Dataset Loading Done.", len(testdataloader))

state_dict = torch.load('/home/leena/pointnet/pointnet.pytorch/utils/seg_0524/seg_model_Bokha_49.pth')
classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval().cuda()
print("Model Loading Done.")

# source = "/home/leena/pointnet/pointnet.pytorch/result/0526"
done =[]
# for filename in os.listdir(source):
#     if filename.endswith('.pts'):
#         name = filename.rstrip(".pts")
#         done.append(name)

shape_ious = []
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target, path = data
    if path[0] in done:
        print("PASS")
        pass
    else:
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points) # [1,2500, 5]
        pred = pred.view(-1, 5) # [2500, 5]
        pred_choice = pred.data.max(1)[1] # [2500]

        file = "/home/leena/pointnet/pointnet.pytorch/result/m_0524/"+path[0]+".pts"
        with open(file, 'w') as file1:
                for label in pred_choice:
                    file1.write(str(label.item())+"\n")
        
        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy()

        for shape_idx in range(target_np.shape[0]):
            parts = range(5)    #np.unique(target_np[shape_idx])
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))
        
        print(path[0], "IoU:",np.mean(part_ious), " Done")
        torch.cuda.empty_cache()

print("mIOU for class Bokha: {}".format(np.mean(shape_ious)))
