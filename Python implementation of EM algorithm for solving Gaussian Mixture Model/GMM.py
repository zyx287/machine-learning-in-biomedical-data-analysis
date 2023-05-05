from numpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 高斯分布
def Gaussian_prob(x,mu,sigma):
    return 1 / ((2 * np.pi) * pow(np.linalg.det(sigma), 0.5)) * np.exp(
        -0.5 * (x - mu).dot(np.linalg.pinv(sigma)).dot((x - mu).T))
# EM算法
def EM(dataMat, maxIter=70):
    m,n=shape(dataMat)
    # 参数初始化
    alpha=[1/4,1/4,1/4,1/4]
    mu=dataMat[np.random.choice(range(m),4)]#随机初始化
    sigma=[mat([[0.1, 0], [0, 0.1]]) for x in range(4)]
    gamma=mat(zeros((m, 4)))
    #EM
    for i in range(maxIter):
        #E
        for j in range(m):
            sum_alpha_gaussian=0
            for k in range(4):
                gamma[j,k]=alpha[k]*Gaussian_prob(dataMat[j,:],mu[k],sigma[k])
                sum_alpha_gaussian+=gamma[j,k]
            for k in range(4):
                gamma[j,k]/=sum_alpha_gaussian
        sumGamma=sum(gamma,axis=0)
        #M
        for k in range(4):
            mu[k]=mat(zeros((1,n)))
            sigma[k]=mat(zeros((n,n)))
            for j in range(m):
                mu[k]+=gamma[j,k]*dataMat[j,:]
            mu[k]/=sumGamma[0,k]
            for j in range(m):
                sigma[k]+=gamma[j,k]*(dataMat[j,:]-mu[k]).T*(dataMat[j,:]-mu[k])
            sigma[k]/=sumGamma[0,k]
            alpha[k]=sumGamma[0,k]/m
    return gamma
#初始随机选择聚类中心点
def random_select_center(dataMat, k):
    numSamples, dim = dataMat.shape
    center = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        center[i, :] = dataMat[index, :]
    return center
#总过程
def GMM_Cluster(dataMat):
    m, n = shape(dataMat)
    center = random_select_center(dataMat, m)
    result = mat(zeros((m, 2)))
    gamma = EM(dataMat)
    for i in range(m):
        result[i, :] = argmax(gamma[i, :]), amax(gamma[i, :])
    for j in range(m):
        pointsInCluster = dataMat[nonzero(result[:, 0].A == j)[0]]
        center[j, :] = mean(pointsInCluster, axis=0)
    return center, result
#画图
def showCluster(dataMat, k, center, result):
    numSamples, dim = dataMat.shape
    mark = ['or', 'ob', 'og', 'ok']
    for i in range(numSamples):
        markIndex = int(result[i, 0])
        plt.plot(dataMat[i, 0], dataMat[i, 1], mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk']
    for i in range(k):
        plt.plot(center[i, 0], center[i, 1], mark[i], markersize=12)
    plt.xlabel('Data[0]')
    plt.ylabel('Data[1]')
    plt.title('The Result Graph of Gaussian Mixture Models for clustering.')
    plt.show()
###
if __name__ == "__main__":
    data = pd.read_csv('D:\\anaconda\\ANACONDA\\envs\\test\\datasetgmm.csv', header=None)
    data = data.values
    dataMat = np.mat(data)
    center, clusterAssign = GMM_Cluster(dataMat)
    print(clusterAssign)
    showCluster(dataMat, 4, center, clusterAssign)
    true_result=pd.read_csv('D:\\anaconda\\ANACONDA\\envs\\test\\resultgmm.csv',header=None)
    true_result=true_result.values
    number_sample=np.shape(dataMat)[0]
    error=0
    for i in range(number_sample):
        if int(clusterAssign[i,0])==int(true_result[i]):
            error+=1
    print(error)