import os
import numpy as np


def scan_dataset(filepath):
    fname = [(x, int(os.path.splitext(x)[0].split('_')[1])) for x in os.listdir(path=filepath)]
    fname.sort(key=lambda x: x[1])
    f_sorted = [filepath+x[0] for x in fname]
    label = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,13,15,16,17,18])
    labels = np.tile(label,len(f_sorted)//19)
    return f_sorted,labels


if __name__ == "__main__":
    path = "./dataset/train_large"
    flist,label = scan_dataset(path)
    print(flist)
    print(label)
