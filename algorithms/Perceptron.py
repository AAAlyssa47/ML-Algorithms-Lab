import numpy as np
class Perceptron:
    """感知机 原始形式"""
    def __init__(self,lr=1.0,max_iter=1000):
        self.lr=lr
        self.max_iter=max_iter
        self.w=None
        self.b=None

    def fit(self,X,y):
        #X(N,n)样本矩阵
        #y(N,)标签{-1，+1}
        N,n=np.shape(X)
        self.w=np.zeros(n,)
        self.b=0.0
        for epoch in range(self.max_iter):
            wrong_count=0
            for i in range(N):
                xi=X[i]
                yi=y[i]
                #判断是否是误分类点
                if yi*(np.dot(self.w,xi)+self.b)<=0:
                    self.w+=self.lr*yi*xi
                    self.b+=self.lr*yi
                    wrong_count+=1
            if wrong_count==0:
                print(f"【原始形式】迭代 {epoch+1} 轮，无错分点，收敛！")
                break
        #达到最大迭代次数，还是有误分类的点
        else:
            print("【原始形式】达到最大迭代次数，未收敛")

    def predict(self,X):
        return np.where(np.dot(X,self.w)+self.b>0,1,-1)

class PerceptronDual:
    """感知机 对偶形式"""
    def __init__(self,lr=1.0,max_iter=1000):
        self.lr=lr
        self.max_iter=max_iter
        self.alpha=None
        self.b=None
        self.Gram=None
        self.X_train=None
        self.y_train=None#保存训练数据，predict就不要手动导入啦

    def fit(self,X,y):
        N,n=np.shape(X)
        self.alpha=np.zeros(N)
        self.b=0.0
        self.X_train=X.copy()
        self.y_train=y.copy()
        self.Gram=X @ X.T

        for epoch in range(self.max_iter):
            wrong_count=0
            for i in range(N):
                yi=y[i]
                pred=np.sum(self.alpha*y*self.Gram[:,i])+self.b
                if yi*pred<=0:
                    self.alpha[i]+=self.lr
                    self.b+=self.lr*yi
                    wrong_count+=1
            if wrong_count==0:
                print(f"【对偶形式】迭代 {epoch+1} 轮，无错分点，收敛！")
                break
        #达到最大迭代次数，还是有误分类的点
        else:
            print("【对偶形式】达到最大迭代次数，未收敛")

    def predict(self,X_test):
        inner=self.X_train @ X_test.T
        pred=self.alpha @ (self.y_train.reshape(-1,1)*inner)+self.b
        
        return np.where(pred>0,1,-1)
    



    
