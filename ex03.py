#!/usr/bin/env python
# coding: utf-8

# # 3. Analytic Geometry  
# ## Manhattam Norm ($l_1$ norm)  
# ### $ ||x||_1 = \Sigma_{i=1}^{n} |x_i| $  
# ## Euclidean Norm ($l_2$ norm)  
# ### $ ||x||_2 = \sqrt{ \Sigma_{i=1}^{n} {x_i}^2 } $  

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams[ "figure.figsize" ] = (10,10) # plot 사이즈 조절


# In[2]:


#1. l2norm 그리기
xRight = np.linspace(0,1,50) # 0 ~ 1
xLeft = np.linspace(-1,0,50) # -1 ~ 0

xarr = [xRight, xLeft, xLeft, xRight] # x coordinate for Q1, Q2, Q3, Q4
xarr = np.array(xarr) # list -> array  (4,50)
xarr = xarr.reshape(-1) # 한 줄로  (200)

yarr = [np.sqrt(1-xRight**2), np.sqrt(1-xLeft**2), -np.sqrt(1-xLeft**2), -np.sqrt(1-xRight**2)]
yarr = np.array(yarr).reshape(-1)

plt.scatter(xarr, yarr, s=.5, color='r')
plt.show()


# In[3]:


1-xRight


# In[4]:


#1. l2norm 그리기
xRight = np.linspace(0,1,50) # 0 ~ 1
xLeft = np.linspace(-1,0,50) # -1 ~ 0

xarr = [xRight, xLeft, xLeft, xRight] # x coordinate for Q1, Q2, Q3, Q4
xarr = np.array(xarr) # list -> array  (4,50)
xarr = xarr.reshape(-1) # 한 줄로  (200)

yarr = [np.sqrt(1-xRight**2), np.sqrt(1-xLeft**2), -np.sqrt(1-xLeft**2), -np.sqrt(1-xRight**2)]
yarr = np.array(yarr).reshape(-1)

plt.scatter(xarr, yarr, s=.5, color='r')

# 2. l1norm 그리기
xarr = [xRight, xLeft, xLeft, xRight] # x coordinate for Q1, Q2, Q3, Q4
xarr = np.array(xarr) # list -> array  (4,50)
xarr = xarr.reshape(-1) # 한 줄로  (200)

yarr = [1-xRight, 1+xLeft, -(1+xLeft), -(1-xRight)]
yarr = np.array(yarr).reshape(-1)

plt.scatter(xarr, yarr, s=.5, color='b')
plt.title('Manhattan Norm(L1, blue), Euclidean Norm(L2, red)')
plt.legend(["l2", "l1"])
plt.show()

