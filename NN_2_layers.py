import os
import numpy as np 
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from glob import iglob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import itertools
from matplotlib.colors import ListedColormap

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
        plt.figure(0)
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
    # plt.figure(4)
    # plt.scatter(fruits_input[:,0],fruits_input[:,1],c=label_clr)
    # plt.xlabel('PCA component 1')
    # plt.ylabel('PCA component 2')
    return fruits_input,one_hot_fruits_output

class dlnet:
    def __init__(self, x, y):
        self.X=x
        self.Y=y
        self.Yh=np.zeros((1,self.Y.shape[0]))
        self.L=2
        self.dims = [2, 100, 3]
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []
        self.lr=0.003
        self.sam = self.Y.shape[0]
        
def nInit(self):    
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]).T / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1]))        
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]).T / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2]))                
        return
    
def Sigmoid(Z):
    return 1/(1+np.exp(-Z))

def dSigmoid(Z):
    return Sigmoid(Z) * (1-Sigmoid (Z))

def Softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(self):    
    
    Z1 = np.dot(self.X,self.param['W1'])+self.param['b1']
    A1 = Sigmoid(Z1)
    self.ch['Z1'],self.ch['A1']=Z1,A1
    
    Z2 = np.dot(A1,self.param['W2']) + self.param['b2'] 
    A2 = Softmax(Z2)
    self.ch['Z2'],self.ch['A2']=Z2,A2
    self.Yh=A2
    #loss=nloss(self,A2)
    return self.Yh#, loss

def nloss(self,Yh):
    loss = Yh - self.Y
    
    #loss = (1./self.dims[2]) * (-np.dot(self.Y,np.log(Yh)) - np.dot(1-self.Y, np.log(1-Yh)))    
    return loss
    

def backward(self):
    dcost_dz2 = self.Yh - self.Y
    dz2_dw2 = self.ch['A1']
    dcost_w2 = np.dot(dz2_dw2.T, dcost_dz2)
    dcost_b2 = dcost_dz2

    ########## Phases 2

    dz2_da1 = self.param['W2']
    dcost_da1 = np.dot(dcost_dz2 , dz2_da1.T)
    da1_dz1 = dSigmoid(self.ch['Z1'])
    dz1_dw1 = self.X
    dcost_w1 = np.dot(dz1_dw1.T, da1_dz1 * dcost_da1)
    dcost_b1 = dcost_da1 * da1_dz1

    # Update Weights ================
    self.param["W1"] -= self.lr * dcost_w1
    self.param["b1"] -= self.lr * dcost_b1.sum(axis=0)  
    self.param["W2"] -= self.lr * dcost_w2   
    self.param["b2"] -= self.lr * dcost_b2.sum(axis=0)  
    
    ''' dLoss_Yh = - (np.divide(self.Y.T, self.Yh ) - np.divide(1 - self.Y.T, 1 - self.Yh))    
        
        dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])    
        dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
        dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
        dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
                            
        dLoss_Z1 = dLoss_A1 * dSigmoid(self.ch['Z1'])        
        dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
        dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X)
        dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  
        
        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
        '''

    
def gd(self,X, Y, iter = 3000):
        np.random.seed(1)                         
    
        nInit(self)
        i=0
        dloss = 9999
        #for i in range(0, iter):
        try:
            while dloss>0:  
                Yh=forward(self)
                backward(self)
            
                if i % 500 == 0:
                    dloss = np.sum(-self.Y * np.log(self.Yh))
                    print ("Cost after iteration %i: %f " %(i, dloss))
                    #self.loss.append(loss)
                i += 1
        except  KeyboardInterrupt:
            print("Stopping training.....")
            return


def pred(self,x, y):  
        self.X=x
        self.Y=y
        pred= forward(self)    
        print("\n\nPREDICTED OUTPUT")
        print(pred)
        comm = np.zeros((y.shape))
        pred_val = np.zeros((y.shape[0]))
        exp_output = np.zeros((y.shape[0]))
        for i in range(0, pred.shape[0]):
            pred_val[i] = pred[i,:].argmax()
            exp_output[i] = y[i,:].argmax()
            comm[i,int(pred_val[i])] = 1
    
        print("\n\nAccuracy : {:.1%}  " .format(np.sum((exp_output == pred_val)/x.shape[0] )))
        
        return pred,comm,pred_val,exp_output
    
def plot_confusion(expected,predicted):
    cf =confusion_matrix(expected,predicted)
    plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    tick_marks = np.arange(len(set(expected))) # length of classes
    class_labels = ['Lychee','Carambula','Pear']
    tick_marks
    plt.xticks(tick_marks,class_labels)
    plt.yticks(tick_marks,class_labels)
    # plotting text value inside cells
    thresh = cf.max() / 2.
    for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
        plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
    plt.show();

def plot_decision_regions(X, y, self, pr_ou , resolution=0.02):

    # setup marker generator and color map
   
    colors = ('green', 'red', 'blue', 'gray', 'cyan')
    

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    #Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    self.X = np.array([xx1.ravel(), xx2.ravel()]).T
    # self.Yh = yh
    # self.Y= y
    Z = forward(self) 
   
    
    comm = np.zeros((Z.shape[0]))
    for i in range(0, Z.shape[0]):
        comm[i] = Z[i,:].argmax()
    Z = comm.reshape(xx1.shape)     
       
    cmap = ListedColormap(colors[:len(np.unique(comm))])
    
    plt.figure(2)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.contour(xx1, xx2, Z,colors = 'k',linewidths = 0.5)
    
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    pos = np.where(pr_ou[:]==0)
    ax = plt.scatter(X[pos,0],X[pos,1],c = 'green',label= 'Lychee')
    bx = plt.scatter(X[np.where(pr_ou[:]==1),0],X[np.where(pr_ou[:]==1),1],c = 'red',label= 'Carambula')
    cx = plt.scatter(X[np.where(pr_ou[:]==2),0],X[np.where(pr_ou[:]==2),1],c = 'blue',label= 'Pear')
    plt.legend(handles = [ax,bx,cx])
    plt.title('Predicted decision region')
    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.show

   
    
[fruits_input,fruits_output] = preprocess('train')
nn = dlnet(fruits_input,fruits_output)
gd(nn,fruits_input,fruits_output, iter = 15000)
[test_input,test_output] = preprocess('test')
pred_train,pred_train_encod,pred_output,act_output = pred(nn,test_input, test_output)
plt.figure(0)
plot_confusion(act_output,pred_output)

plt.figure(1)
plt.subplot(121)
plt.scatter(test_input[:,0],test_input[:,1],c=test_output)
plt.title('Actual')
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
plt.subplot(122)
plt.scatter(test_input[:,0],test_input[:,1],c=pred_train_encod)
plt.title('Predicted')
plt.xlabel('PCA component 1')
#plt.ylabel('PCA component 2')
plt.show


plot_decision_regions(test_input,pred_train_encod,nn,pred_output)
