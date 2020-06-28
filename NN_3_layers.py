import os
import numpy as np 
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from glob import iglob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import itertools

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
    #plt.scatter(fruits_input[:,0],fruits_input[:,1],c=label_clr)
    
    return fruits_input,one_hot_fruits_output

class dlnet:
    def __init__(self, x, y):
        self.X=x
        self.Y=y
        self.Yh=np.zeros((1,self.Y.shape[0]))
        self.L=2
        self.dims = [2, 4, 4, 3]
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []
        self.lr=0.003
        self.sam = self.Y.shape[0]
        
def nInit(self):    
        np.random.seed(1)
        self.param['Wh1'] = np.random.randn(self.dims[1], self.dims[0]).T / np.sqrt(self.dims[0]) 
        self.param['bh1'] = np.zeros((self.dims[1]))        
        self.param['Wh2'] = np.random.randn(self.dims[2], self.dims[1]).T / np.sqrt(self.dims[1]) 
        self.param['bh2'] = np.zeros((self.dims[2]))     
        self.param['Wo'] = np.random.randn(self.dims[3], self.dims[2]).T / np.sqrt(self.dims[2]) 
        self.param['bo'] = np.zeros((self.dims[3]))           
        return
    
def Sigmoid(Z):
    return 1/(1+np.exp(-Z))

def dSigmoid(Z):
    return Sigmoid(Z) * (1-Sigmoid (Z))

def Softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(self):    
    
    Zh1 = np.dot(self.X,self.param['Wh1'])+self.param['bh1']
    Ah1 = Sigmoid(Zh1)
    self.ch['Zh1'],self.ch['Ah1']=Zh1,Ah1
    
    Zh2 = np.dot(Ah1,self.param['Wh2']) + self.param['bh2'] 
    Ah2 = Softmax(Zh2)
    self.ch['Zh2'],self.ch['Ah2']=Zh2,Ah2
    
    Zo = np.dot(Ah2,self.param['Wo']) + self.param['bo'] 
    Ao = Softmax(Zo)
    self.ch['Zo'],self.ch['Ao']=Zh2,Ah2
    
    self.Yh=Ao
    loss=nloss(self,Ao)
    return self.Yh, loss

def nloss(self,Yh):
    loss = Yh - self.Y
    
    #loss = (1./self.dims[2]) * (-np.dot(self.Y,np.log(Yh)) - np.dot(1-self.Y, np.log(1-Yh)))    
    return loss
    

def backward(self):
    dcost_dzo = self.Yh - self.Y
    dzo_dwo = self.ch['Ah2']
    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)
    dcost_bo = dcost_dzo

########## Phases 2

    dzo_dah2 = self.param['Wo']
    dcost_dah2 = np.dot(dcost_dzo , dzo_dah2.T)
    dah2_dzh2 = dSigmoid(self.ch['Zh2'])
    dzh2_dwh2 = self.ch['Ah1']
    dcost_wh2 = np.dot(dzh2_dwh2.T, dah2_dzh2 * dcost_dah2)
    dcost_bh2 = dcost_dah2 * dah2_dzh2
    
    ########## Phases 3

    #dzo_dah2 = self.param['Wo']
    #dcost_dah2 = np.dot(dcost_dzo , dzo_dah2.T)
    #dah2_dzh2 = dSigmoid(self.ch['Zh2'])
    dzh2_dah1 = self.param['Wh2']
    dah1_dzh1 = dSigmoid(self.ch['Zh1'])
    dzh1_dwh1 = self.X
#    dzh2_dwh1 = np.dot(dzh1_dwh1.T,np.dot(dzh2_dah1,dah1_dzh1.T).T)
   # dcost_dzh2=  dah2_dzh2 * dcost_dah2
    #dcost_dwh1 = np.dot(dzh2_dwh1,dcost_dzh2.T)
    dcost_bh1 = np.dot((dah2_dzh2 * dcost_dah2).T, np.dot(dzh2_dah1,dah1_dzh1.T).T)
    dcost_dwh1 = np.dot(((dah2_dzh2 * dcost_dah2).T * np.dot(dzh2_dah1,dah1_dzh1.T)),dzh1_dwh1).T
    
    # Update Weights ================
    self.param["Wh1"] -= self.lr * dcost_dwh1
    self.param["bh1"] -= self.lr * dcost_bh1.sum(axis=0)  
    self.param["Wh2"] -= self.lr * dcost_wh2   
    self.param["bh2"] -= self.lr * dcost_bh2.sum(axis=0)  
    self.param["Wo"] -= self.lr * dcost_wo   
    self.param["bo"] -= self.lr * dcost_bo.sum(axis=0)  
    
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
                Yh, loss=forward(self)
                backward(self)
            
                if i % 500 == 0:
                    dloss = np.sum(-self.Y * np.log(self.Yh))
                    print ("Cost after iteration %i: %f " %(i, dloss))
                    self.loss.append(loss)
                i += 1
        except  KeyboardInterrupt:
            print("Stopping training.....")
            return


def pred(self,x, y):  
        self.X=x
        self.Y=y
        pred, loss= forward(self)    
        comm = np.zeros((y.shape))
        print("\n\nPREDICTED OUTPUT")
        print(pred)
        pred_val = np.zeros((y.shape[0]))
        exp_output = np.zeros((y.shape[0]))
        for i in range(0, pred.shape[0]):
            pred_val[i] = pred[i,:].argmax()
            exp_output[i] = y[i,:].argmax()
            comm[i,int(pred_val[i])] = 1
    
        print("\n\nAccuracy : {:.1%}  " .format(np.sum((exp_output == pred_val)/x.shape[0] )))
        
        return pred,comm,pred_val,exp_output
    
def plot_confusion(expected,predicted):
    plt.figure(0)
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