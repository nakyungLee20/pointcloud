from __future__ import print_function
#from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data as data
from torch.autograd import Variable
# from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls
import matplotlib.pyplot as plt
import json
import time
import os

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='/home/leena/pointnet/pointnet.pytorch/utils/seg_final/seg_model_Bokha_46.pth', help='model path')
parser.add_argument('--idx', type=int, default=1, help='model index')
parser.add_argument('--dataset', type=str, default='/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0', help='dataset path')
parser.add_argument('--class_choice', type=str, default='BokhaTest', help='class choice')

opt = parser.parse_args()
print(opt)

output_dir = '/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/0216/points_label'
roots = '/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/0216/points'

# Load dataset 
class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
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
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        # splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        splitfile = os.path.join(self.root, 'train_test_split', '{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

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
        #seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)
        
        #resample
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        point_set = torch.from_numpy(point_set)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls, fn
        else:
            return point_set, fn

    def __len__(self):
        return len(self.datapath) # number of datas in one category

datasets = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(
    datasets,
    batch_size=1,
    shuffle=True,
    num_workers=4)

print("Dataset Loading Done.")
# with open('/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/train_test_split/test_file_list.json', 'r') as f:
#     testlist = json.load(f)

# test_file = []
# for li in testlist:
#     li = li.lstrip('shape_data/0216')
#     file = root + li
#     test_file.append(file)

# for test in test_file:
#     point_set = np.loadtxt(test).astype(np.float32)

# Load model
state_dict = torch.load(opt.model)
classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval().cuda()
print("Model Loading Done.")

# Inference
num_batch = len(datasets) / 4
undo =[]
results = {}
start = time.time()

for i, data in enumerate(testdataloader, 0):
    point, path = data
    path = path[1][0].split('0216/points/')[1].rstrip('.pts')
    point = point.transpose(2, 1)
    points = point.cuda()
    # print(points.shape)
    
    try: 
        pred, _, _ = classifier(points)
        pred = pred.view(-1, 9)
        pred_choice = pred.data.max(1)[1]
        
        results[path] = pred_choice
        
        file = output_dir + '/' + path + ".seg"
        with open(file, 'w') as file:
            for label in pred_choice:
                file.write(str(label.item())+"\n")
            # file.close()
        
        torch.cuda.empty_cache()
        print(f'Inference {path} done.')
        
    except:
        undo.append(path)
        print(f'Inference {path} error.')

sec = time.time() - start
print("Inference Time:", sec)
print(undo)

# for res in results:
#     print(res)
#     file = output_dir+'/'+list(res.keys())[0]+".seg"
#     with open(file, 'w') as file:
#         for label in res.values():
#             file.write(label+"\n")
