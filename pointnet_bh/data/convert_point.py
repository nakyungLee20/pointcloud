import laspy

las = laspy.read("bokha_01_classified.las")

x_dim = las.x
y_dim = las.y
z_dim = las.z
labels = las.classification

#point_format = las.point_format
#print(list(point_format.dimension_names))


with open("bh_01.pts", "w") as pts_file:
    for x, y, z in zip(x_dim, y_dim, z_dim):
        x = int(x)
        y = int(y)
        z = int(z)
        pts_file.write(f"{x} {y} {z}\n")
        
#with open("bh_01_label.pts", "w") as pts_file:
#    for x, label in zip(x_dim, labels):
#        x = int(x)
#        label = int(label)
#        pts_file.write(f"{label}\n")

print("Conversion completed. Output saved as output.pts")

