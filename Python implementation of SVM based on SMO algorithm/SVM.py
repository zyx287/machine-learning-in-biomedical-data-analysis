import numpy as np
import pandas as pd
import copy
import csv
import random
from random import uniform
def datapreparation(filename):
    with open(filename,'r+') as csvfile:
        lines=csv.reader(csvfile)
        data_set=list(lines)
        numb=len(data_set[0])-1
    for data in data_set:
        for i in range(numb):
            if data[i] != '':
                data[i]=float(data[i])
            else:
                data[i]=np.NaN
        data[numb]=int(data[numb])
    dt_pre=pd.DataFrame(data_set)
    res=dt_pre.dropna(axis=0, how='any')
    data_array=np.array(res)
    result=data_array.tolist()
    for data in result:
        data[numb]=int(data[numb])
    return result
##将X与Y分开
def separation(dataname):
    data_pre=copy.deepcopy(dataname)
    data_mat=[]
    label_mat=[]
    num = len(data_pre[0]) - 1
    for i in range(len(data_pre)):
        if data_pre[i][num]==0:
            data_pre[i][num]=-1
        label_mat.append(data_pre[i][num])
        del(data_pre[i][num])
        data_mat.append(data_pre[i])
    # print(label_mat)
    data_mat=np.mat(data_mat)
    label_mat=np.mat(label_mat).transpose()
    return data_mat, label_mat
###SVM
##核函数
def kernel(X,Sample,k_info):
    m,n=np.shape(X)
    K_mat=np.mat(np.zeros((m,1)))
    #linear
    if k_info[0]=="linear":
        K_mat=X*(Sample.transpose())
    #polynomial
    elif k_info[0]=="poly":
        K_mat=X*(Sample.transpose())
        for i in range(m):
            K_mat[i]=K_mat[i]**k_info[1]
    #Radial basis
    elif k_info[0]=="rbf":
        for i in range(m):
            delta=X[i,:]-Sample
            K_mat[i]=delta*delta.transpose()
        K_mat=np.exp(K_mat/(-1*k_info[1]**2))
    else:
        raise NameError("核函数输入错误")
    return K_mat
#随机选取aj
def aj_ranselect(i,m):
    j=i
    while j==i:
        j=int(random.uniform(0,m))
    return j
##SVM的类
class SVM:
    def __init__(self,train_data,train_label,C,toler,k_info):
        self.train_data=X
        self.train_label=Y
        self.C=C
        self.toler=toler
        self.m=np.shape(X)[0]#样本数m
        self.n=np.shape(X)[1]#特征数n
        self.alpha=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.E=np.mat(np.zeros((self.m,2)))
        self.K=np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernel(self.train_data,self.train_data[i,:],k_info)
##计算残差
def calEi(svm,i):
    fxi = float(np.multiply(svm.alpha,svm.train_label).T * svm.K[:, i] + svm.b)
    Ei = fxi - float(svm.train_label[i])
    return Ei
##选择违反KKT条件且误差最大的aj
def selectJKTT(i,svm,Ei):
    maxK=-1
    maxdeltaE=0
    Ej=0
    svm.E[i]=[1,Ei]
    validKTT=np.nonzero(svm.E[:,0].A)[0]
    # print(validKTT)
    if (len(validKTT))>1:
        for k in validKTT:
            if k==i:
                continue
            Ek=calEi(svm,k)
            deltaE=abs(Ei-Ek)
            if (deltaE>maxdeltaE):
                maxK=k
                maxdeltaE=deltaE
                Ej=Ek
        return maxK,Ej
    else:
        j=aj_ranselect(i,svm.m)
        Ej=calEi(svm,j)
    return j,Ej
##用新的alpha计算残差
def updateE(svm,k):
    Ek=calEi(svm,k)
    svm.E[k]=[1,Ek]
##计算迭代次数
def smo_pre(svm,i):
    Ei=calEi(svm,i)
    if ((svm.train_label[i]*Ei<-svm.toler) and (svm.alpha[i]<svm.C)) or ((svm.train_label[i]*Ei>svm.toler) and (svm.alpha[i]>0)):
        j,Ej=selectJKTT(i,svm,Ei)
        alpha1=svm.alpha[i]
        alpha2=svm.alpha[j]
        y1, y2 = svm.train_label[i], svm.train_label[j]
        s = y1 * y2
        if y1 == y2:
            L = max(0, alpha1 + alpha2 - svm.C)
            H = min(svm.C, alpha1 + alpha2)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(svm.C, svm.C + alpha2 - alpha1)
        if L==H:
            return 0
        K11 = svm.K[i,i]
        K12 = svm.K[i,j]
        K22 = svm.K[j,j]
        eta=K11+K22-2.0*K12
        if eta<=0:
            return 0
        a2=alpha2+y2*(Ei-Ej)/eta
        a2 = min(a2, H)  # clip
        a2 = max(a2, L)  # clip
        svm.alpha[j]=a2
        updateE(svm,j)
        if (abs(svm.alpha[j]-alpha2)<0.00001):#更新不大了
            return 0
        a1=alpha1+s*(alpha2-a2)
        svm.alpha[i]=a1
        updateE(svm,i)
        b = svm.b
        b1 = y1 * K11 * (alpha1 - a1) + y2 * K12 * (alpha2 - a2) + b - Ei
        b2 = y1 * K12 * (alpha1 - a1) + y2 * K22 * (alpha2 - a2) + b - Ej
        b_new = (b1 + b2) * 0.5
        svm.b=b_new
        return 1
    else:
        return 0
#SMO算算子和beta0
def smo(X, Y, C, toler, maxIter, k_info=("rbf", 2)):
    svm=SVM(np.mat(X),np.mat(Y),C,toler,k_info)
    iter = 0
    entireSet = True
    alpha_change = 0
    m=np.shape(X)[0]
    while (iter < maxIter) and (alpha_change > 0) or entireSet:
        alpha_change = 0
        if entireSet:
            for i in range(m):
                alpha_change += smo_pre(svm, i)
            iter += 1
        else:
            nonbound = np.nonzero((svm.alpha.A > 0) * (svm.alpha.A < 0))[0]
            for i in nonbound:
                alpha_change += smo_pre(svm, i)
            iter += 1
        if entireSet:
            entireSet = False
        elif (alpha_change == 0):
            entireSet = True
    return svm.b, svm.alpha
if __name__ == "__main__":#高斯rbf
    C=1
    tol = 0.001
    maxIter = 100
    sigma=2
    kernelinfo = "rbf"
    data_train = datapreparation("D:\\anaconda\\ANACONDA\\envs\\test\\gpl96.csv")
    X, Y = separation(data_train)
    b, alpha = smo(X, Y, C, tol, maxIter, (kernelinfo, sigma))  # X,Y,C,tol,maxIter
    # print(b, alpha.T)
    svInd = np.nonzero(alpha.A > 0)[0]
    sVs = X[svInd]
    labelSV = Y[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    errorCount = 0
    m, n = np.shape(X)
    for i in range(m):
        kernelEval = kernel(sVs, X[i, :], (kernelinfo, sigma))
        predict = kernelEval.T * np.multiply(labelSV, alpha[svInd]) + b
        if np.sign(predict) != np.sign(Y[i]):
            errorCount += 1
        acc = 1.0 - (float(errorCount) / m)
    # print("training acc: %f" % acc)
    data_test = datapreparation("D:\\anaconda\\ANACONDA\\envs\\test\\gpl97.csv")
    Xtest, Ytest = separation(data_test)
    errorCount = 0
    for i in range(m):
        kernelEval = kernel(sVs, Xtest[i, :], (kernelinfo, sigma))
        predict = kernelEval.T * np.multiply(labelSV, alpha[svInd]) + b
        if np.sign(predict) != np.sign(Ytest[i]):
            errorCount += 1
    acc = 1.0 - (float(errorCount) / m)
    print("test acc: %f" % acc)
if __name__ == "__main__":#linear
    C = 1
    tol = 0.001
    maxIter = 100
    kernelinfo = "linear"
    data_train = datapreparation("D:\\anaconda\\ANACONDA\\envs\\test\\gpl96.csv")
    X, Y = separation(data_train)
    b, alpha = smo(X, Y, C, tol, maxIter, (kernelinfo, sigma))  # X,Y,C,tol,maxIter
    svInd = np.nonzero(alpha.A > 0)[0]
    sVs = X[svInd]
    labelSV = Y[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    errorCount = 0
    m, n = np.shape(X)
    for i in range(m):
        kernelEval = kernel(sVs, X[i, :], (kernelinfo, sigma))
        predict = kernelEval.T * np.multiply(labelSV, alpha[svInd]) + b
        if np.sign(predict) != np.sign(Y[i]):
            errorCount += 1
        acc = 1.0 - (float(errorCount) / m)
    # print("training acc: %f" % acc)
    data_test = datapreparation("D:\\anaconda\\ANACONDA\\envs\\test\\gpl97.csv")
    Xtest, Ytest = separation(data_test)
    errorCount = 0
    for i in range(m):
        kernelEval = kernel(sVs, Xtest[i, :], (kernelinfo, sigma))
        predict = kernelEval.T * np.multiply(labelSV, alpha[svInd]) + b
        if np.sign(predict) != np.sign(Ytest[i]):
            errorCount += 1
        acc = 1.0 - (float(errorCount) / m)
    print("test acc: %f" % acc)










