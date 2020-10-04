#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

x = np.array([[1,2,3],[4,5,6]])
print "x:\n", x                   


# In[9]:


import matplotlib.pyplot as plt

# -10에서 10까지 100개의 간격으로 나뉘어진 배열을 생성
x = np.linspace(-10,10,100)
# sin 함수를 사용하여 y배열을 생성
y = np.sin(x)
# plot 함수는 한 배열의 값을 다른 배열에 대응해서 선 그래프를 그림
plt.plot(x,y,marker='x')
plt.show()


# In[3]:


from scipy import sparse

# 대각선 원소는 1이고 나머지는 0인 2차원 NumPy 배열을 만듦.
eye = np.eye(4)
print "NumPy 배열:\n", eye


# In[4]:


from scipy import sparse
# NumPy 배열을 CSR 포맷의 SciPy 희소 행렬로 변환
# 0이 아닌 원소만 저장됨
# (참고) CSR: Compressed sparse row. 행의 인덱스를 압축하여 저장
sparse_matrix = sparse.csr_matrix(eye)
print "SciPy의 CSR 행렬: \n", sparse_matrix


# In[5]:


data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print "COO 표현:\n", eye_coo


# In[6]:


import pandas as pd

# 회원 정보가 들어간 간단한 데이터셋을 생성
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location': ["New York", "Paris", "Berlin", "London"],
        'Age': [24, 13, 53, 33]
       }

data_pandas = pd.DataFrame(data)

# IPython.display는 주피터 노트북에서 DataFrame을 미려하게 출력해줌
from IPython.display import display
display(data_pandas)
# print(data_pandas)


# In[7]:


# Age 열의 값이 30 이상인 모든 행을 선택함
display(data_pandas[data_pandas.Age > 30])


# In[8]:


import sys
print "Python version:", sys.version

import pandas as pd
print "pandas version:", pd.__version__

import matplotlib
print "matplotlib version:", matplotlib.__version__

import numpy as np
print "NumPy 버전:", np.__version__

import scipy as sp
print "SciPy 버전:", sp.__version__

import IPython
print "IPython 버전:", IPython.__version__

import sklearn
print "scikit-learn 버전:", sklearn.__version__

