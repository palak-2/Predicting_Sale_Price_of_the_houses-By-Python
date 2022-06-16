#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[2]:


a=pd.read_csv('Transformed_Housing_Data2.csv')


# In[4]:


#Random intilization y is dependent or target variable
def para_in(y):
    m=0.1
    c=y.mean()
    return m,c


# In[42]:


#Genrate prediction here x=independent variable
def gen_pred(m,c,x):
    pred=[]
    for i in x:
        vals=(m*i)+c
        pred.append(vals)
    return pred


# In[13]:


#calculating cost= mean square error betwwen predicted and actuals
def cal_cost(pred,y):
    cost=np.sum((abs(pred-y))**2/len(y))
    return cost


# In[11]:


#update pARAMETER=1-calculating gradient gm or gc(by partial differentiation),2-update  parameter m and c
#calculate gradient
def cal_grad(pred,y,x):
    n=len(y)
    Gm=2/n*np.sum((pred-y)*x)
    Gc=2/n*np.sum((pred-y))
    return Gm,Gc


# In[12]:


#update parameter
def up_param(m_old,c_old,Gm_old,Gc_old,alpha):
    m_new=m_old-alpha*Gm_old
    c_new=c_old-alpha*Gc_old
    return m_new,c_new


# In[52]:


#final result of gradient decent
def result(m,c,x,y,cost,pred,i):
    # if the gradient decent has converged to the optimumvalue before max iteration
    if i<max_iter-1:
        print("gradient descent has converged at iteration {}--".format(i))
    else:
        print("result after",max_iter,'iteration is:***********')
            
##plotting final result
    plt.figure(figsize=(14,7),dpi=120)
    plt.scatter(x,y,color='red',label='data points')
    label='final regression line: m={},c={}'.format(str(m),str(c))
    plt.plot(x,pred,color='green',label='label')
    plt.xlabel('Flatarea')
    plt.ylabel('sale price')
    plt.show()


# In[29]:


sale_price=a['Sale_Price'].head(30)
flat_area=a['Flat Area (in Sqft)'].head(30)
b=pd.DataFrame({'sale_price': sale_price,'flat_area':flat_area})


# In[31]:


sale_price.shape


# In[66]:


# scaling the data set using the standard scaler as value of m and are very very large
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#defing and reshaping the data set
sale_price=b['sale_price'].values.reshape(-1,1)
flat_area=b['flat_area'].values.reshape(-1,1)

#declaring parameter
max_iter=600
cost_old=0
alpha=0.01

# step 1: intilizing the values
m,c=para_in(sale_price)

##gradient descent in action
for i in range(0,max_iter):
    
    ##generating prediction
    prediction=gen_pred(m,c,flat_area)
    
    #calculating cost
    cost_new=cal_cost(prediction,sale_price)
    
    #checking if GD converged
    if abs(cost_new-cost_old)<10**(-7):
        break
        
    #calculating gradients
    Gm,Gc=cal_grad(prediction,sale_price,flat_area)
    
    #updating parameter m and c
    m,c=up_param(m,c,Gm,Gc,alpha)
    
    #display result after 20 iteration
    if i%20==0:
        print("After iteration",i,": m=",m, ' ; c=',c,'; cost=',cost_new)
    #updating old cost
    cost_old=cost_new
#final result
result(m,c,flat_area,sale_price,cost_new,prediction,i)


# In[35]:





# In[ ]:





# In[ ]:





# In[ ]:




