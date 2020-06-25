import os
import numpy as np 
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from glob import iglob

train_path = os.path.join(os.getcwd(),"Data/Data_train/Lychee")
fruits = pd.DataFrame([])

for path in iglob(os.path.join(train_path,'*.png')):
    
    img = imread(path)
    #plt.imshow(img)

    fruit = pd.Series(img.flatten(),name = os.path.split(path)[1])
    fruits = fruits.append(fruit)

fruits_pca = PCA()
fruits_pca.fit(fruits)

class dlnet:
    def __init__(self, x, y):
        self.X=x
        self.Y=y
        self.Yh=np.zeros((1,self.Y.shape[1]))
        self.L=2
        self.dims = [9, 15, 1]
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []
        self.lr=0.003
        self.sam = self.Y.shape[1]