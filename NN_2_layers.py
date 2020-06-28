import os
import numpy as np 
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from glob import iglob
from sklearn.preprocessing import MinMaxScaler
 
 

def preprocess(dataset):
    if dataset == 'train':
        train_path = os.path.join(os.getcwd(),"Data/Data_train")
        
    if dataset == 'validate':
       train_path = os.path.join(os.getcwd(),"Data/Data_validate")
       if os.path.exists(train_path) == False:
           print("NOT THERE")
    
    if dataset == 'test':
        train_path = os.path.join(os.getcwd(),"Data/Data_test")
        
    fruits = pd.DataFrame([])
    labels = []
    fruits_output = []
    for path in iglob(os.path.join(train_path,'*/*.png')):
        
        label = os.path.dirname(path)
        label = os.path.split(label)[1]
        labels.append(label)
        if label == 'Lychee':
            fruits_output.append(0)
        if label == 'Carambula':
            fruits_output.append(1)
        if label == 'Pear':
            fruits_output.append(2)
        
        img = imread(path)
        #plt.imshow(img[:,:,0])
    
        fruit = pd.Series(img[:,:,0].flatten(),name = label)
        fruits = fruits.append(fruit)
        
   
    fruits_output = np.array(fruits_output)
    
    one_hot_fruits_output= np.zeros((len(fruits_output), 3))

    for i in range(len(fruits_output)):
        one_hot_fruits_output[i, fruits_output[i]] = 1
        
    
    #normalization
    scaler = MinMaxScaler() 
    scaled_fruits = scaler.fit_transform(fruits.iloc[:,0:fruits.shape[1]]) 
    scaled_fruits = pd.DataFrame(scaled_fruits, index = fruits_output)
    
    
    #For scatter plot    
    label_clr = np.array(labels)
    label_clr['Lychee'==label_clr]='red'  
    label_clr['Carambula'==label_clr]='green'
    label_clr['Pear'==label_clr]='blue'  
    
    
    fruits_pca = PCA(n_components=2)
    fruits_input = fruits_pca.fit_transform(scaled_fruits)
    
    plt.scatter(fruits_input[:,0],fruits_input[:,1],c=label_clr)
    return fruits_input,one_hot_fruits_output

class dlnet:
    def __init__(self, x, y):
        self.X=x
        self.Y=y
        self.Yh=np.zeros((1,self.Y.shape[0]))
        self.L=2
        self.dims = [2, 15, 3]
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []
        self.lr=0.003
        self.sam = self.Y.shape[0]
        
def nInit(self):    
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))        
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))                
        return
    
def Sigmoid(Z):
    return 1/(1+np.exp(-Z))

def forward(self):    
        Z1 = self.param['W1'].dot(self.X.transpose()) + self.param['b1'] 
        A1 = Sigmoid(Z1)
        self.ch['Z1'],self.ch['A1']=Z1,A1
        
        Z2 = self.param['W2'].dot(A1) + self.param['b2']  
        A2 = Sigmoid(Z2)
        self.ch['Z2'],self.ch['A2']=Z2,A2
        self.Yh=A2
        loss=nloss(self,A2)
        return self.Yh, loss

def nloss(self,Yh):
        loss = (1./self.sam) * (-np.dot(self.Y,np.log(Yh)) - np.dot(1-self.Y, np.log(1-Yh)))    
        return loss
    
def dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def dSigmoid(Z):
    s = 1/(1+np.exp(-Z))
    dZ = s * (1-s)
    return dZ

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))    
        
        dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])    
        dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
        dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
        dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
                            
        dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])        
        dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
        dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X)
        dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  
        
        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2

def gd(self,X, Y, iter = 3000):
        np.random.seed(1)                         
    
        nInit(self)
    
        for i in range(0, iter):
            Yh, loss=forward(self)
            backward(self)
        
            if i % 500 == 0:
                
                print ("Cost after iteration %i: %f %f %f" %(i, loss[0],loss[1],loss[2]))
                self.loss.append(loss)
    
        return


def pred(self,x, y):  
        self.X=x
        self.Y=y
        comp = np.zeros((1,x.shape[0]))
        pred, loss= forward(self)    
        print(pred)
        for i in range(0, pred.shape[1]):
            if pred[0,i] > 0.5: comp[0,i] = 1
            else: comp[0,i] = 0
    
        print("Acc: " + str(np.sum((comp == y)/x.shape[1])))
        
        return comp
      
[fruits_input,fruits_output] = preprocess('train')
nn = dlnet(fruits_input,fruits_output)
gd(nn,fruits_input,fruits_output, iter = 15000)
[test_input,test_output] = preprocess('test')
pred_train = pred(nn,test_input, test_output)