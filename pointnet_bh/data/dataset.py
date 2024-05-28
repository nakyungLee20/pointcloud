import laspy
import os

#point_format = las.point_format
#print(list(point_format.dimension_names))

def convert(fname):
    las = laspy.read(fname)
    
    x_dim = las.x
    y_dim = las.y
    z_dim = las.z
    labels = las.classification
    
    fwrite = fname.split("Segmented/")[1]
    fwrite = fwrite.strip(".las")
    fpath1 = "/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/0524/points/"+fwrite+".pts"
    fpath2 = "/home/leena/pointnet/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/0524/points_label/"+fwrite+".seg"
    
    with open(fpath1, "w") as pts_file1:
        for x, y, z in zip(x_dim, y_dim, z_dim):
            x = round(float(x), 3)
            y = round(float(y), 3)
            z = round(float(z), 3)
            pts_file1.write(f"{x} {y} {z}\n")

    with open(fpath2, "w") as pts_file2:
        for label in labels:
            label = int(label)
            if label == 4:
                label = 3
            elif label == 5:
                label = 4
            elif label > 5:
                label = 0
            
            pts_file2.write(f"{label}\n")
            
    print(f"Converting {fwrite} file done. ")


# process each .las file 
root = "/home/leena/pointnet/datasets/Segmented"
for filename in os.listdir(root):
    if filename.endswith('.las'):
        las_path = os.path.join(root, filename)
        convert(las_path)
        