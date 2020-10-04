#!/usr/bin/env python
# coding: utf-8

# # 1. 파이썬 기초(Optional)
# ## 프린트

# In[1]:


print ("Hello, world")

# integer
x = 3
print ("정수: %01d, %02d, %03d, %04d, %05d" % (x,x,x,x,x))

# float
x = 256.123
print ("실수: %.0f, %.1f, %.2f" % (x,x,x))

# string
x = "Hello, world"
print ("문자열: [%s]" % (x))


# ## 반복문, 조건문

# In[2]:


contents = ["Regression", "Classification", "SYM", "Clustering", "Demension reduction", "NN", "CNN", "AE", "GAN", "RNN"]

for con in contents:
    if con in ["Regression", "Classification", "SYM", "Clustering", "Demension reduction"]:
        print ("%s 은(는) 기계학습 내용입니다." %con)
    elif con in ["CNN"]:
        print ("%s 은(는) convolutional neural network 입니다." %con)
    else:
        print ("%s 은(는) 심층학습 내용입니다." %con)


# ## 반복문과 인덱스

# In[3]:


for (i,con) in enumerate(contents):
    print ("[%d/%d]: %s" %(i, len(contents), con))


# ## 함수

# In[4]:


def sum(a,b):
    return a+b

x = 10.0
y = 20.0
print ("%.1f + %.1f = %.1f" %(x, y, sum(x,y)))


# ## 리스트

# In[5]:


a = []
b = [1,2,3]
c = ["Hello", "," "world"]
d = [1,2,3,"x","y","z"]
x = []
print (x)

x.append('a')
print (x)

x.append(123)
print (x)

x.append(["a", "b"])
print x


# ## 딕셔너리(dictionary)

# In[6]:


dic = dict()
dic["name"] = "Heekyung"
dic["town"] = "Goyang city"
dic["job"] = "Assistant professor"
print dic


# ## 클래스

# In[7]:


class Student:
    # 생성자
    def __init__(self, name):
        self.name = name
    # 메써드
    def study(self, hard = False):
        if hard:
            print "%s 학생은 열심히 공부합니다." %self.name
        else:
            print "%s 학생은 공부합니다." %self.name
            
s = Student('Heekyung')
s.study()
s.study(hard = True)


# ## 라이브러리(패키지) 로드

# In[8]:


import numpy as np


# # 2. 벡터, 행렬 연산, 그래프 그리기
# ## 프린트

# In[9]:


def print_val(x):
    print "Type:", type(x)
    print "Shape:", x.shape
    print "값:\n", x
    print " "


# ## rank 1 np array

# In[10]:


x = np.array([1,2,3])
print_val(x)

x[0] = 5
print_val(x)


# ## rank 2 np array

# In[11]:


y = np.array([[1,2,3], [4,5,6]])
print_val(y)


# ## rank 2 ones

# In[12]:


a = np.ones((3,2))
print_val(a)


# ## rank 2 zeros

# In[13]:


a = np.zeros((2,2))
print_val(a)


# ## rank 2 단위 행렬(identity matrix)

# In[14]:


a = np.eye(3,3)
print_val(a)


# ## 랜덤 행렬(uniform: 0~1 사이 모든 값들이 나올 확률이 같음)

# In[15]:


a = np.random.random((4,4))
print_val(a)


# ## 랜덤 행렬(Gaussian: 0을 평균으로 하는 가우시안 분포를 따르는 랜덤값)

# In[16]:


a = np.random.randn(4,4)
print_val(a)


# ## np array indexing

# In[17]:


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print_val(a)


# In[18]:


b = a[:2, 1:3] # 행 0~1, 열 1~2
print_val(b)


# ## 행렬의 n번째 행 얻기

# In[19]:


row1 = a[1, :] # 1번째 행
print_val(row1)


# ## 행렬의 원소별 연산

# In[20]:


m1 = np.array([[1,2], [3,4]], dtype = np.float64)
m2 = np.array([[5,6], [7,8]], dtype = np.float64)


# In[21]:


# elementwise sum
print_val(m1 + m2)
print_val(np.add(m1, m2))


# In[22]:


# elementwise difference
print_val(m1 - m2)
print_val(np.subtract(m1, m2))


# In[23]:


# elementwise product
print_val(m1 * m2)
print_val(np.multiply(m1, m2))


# In[24]:


# elementwise division
print_val(m1 / m2)
print_val(np.divide(m1, m2))


# In[25]:


# elementwise square root
print_val(np.sqrt(m1))


# ## 행렬 연산

# In[26]:


m1 = np.array([[1,2], [3,4]]) # (2,2)
m2 = np.array([[5,6], [7,8]]) # (2,2)
v1 = np.array([9,10]) # (2,1) # [[9,10]] (1,2)
v2 = np.array([11,12]) # (2,1)

print_val(m1)
print_val(m2)
print_val(v1)
print_val(v2)


# ## 벡터-벡터 연산

# In[27]:


print_val(v1.dot(v2))
print_val(np.dot(v1, v2))


# ## 벡터-행렬 연산

# In[28]:


print_val(m1.dot(v1)) # (2,2) x (2,1) -> (2,1)
print_val(np.dot(m1,v1))


# ## 행렬-행렬 연산

# In[29]:


print_val(m1.dot(m2))
print_val(np.dot(m1, m2))


# ## 전치 행렬(transpose)

# In[30]:


print_val(m1)
print_val(m1.T)


# ## 합

# In[31]:


print_val(np.sum(m1)) # 행렬의 모든 원소의 합
print_val(np.sum(m1, axis = 0)) # shape[0] (행) 을 압축시키자. (2,2) -> (2,)
print_val(np.sum(m1, axis = 1)) # shape[1] (열) 을 압축시키자. (2,2) -> (2,)


# In[32]:


m1 = np.array([[1,2,3], [4,5,6]])
print m1
print m1.shape # (2,3)


# In[33]:


print np.sum(m1)
print np.sum(m1, axis = 0) # shape[0] (행) 을 압축시키자. (2,3) -> (3,)
print np.sum(m1, axis = 1) # shape[1] (열) 을 압축시키자. (2,3) -> (2,)


# ## zeros-like

# In[34]:


m1 = np.array([[1,2,3],
              [4,5,6],
              [7,8,9],
              [10,11,12]])
m2 = np.zeros_like(m1) # m1과 같은 형태의 0으로 이루어진 np array
print_val(m1)
print_val(m2)


# ## Matplot library

# In[35]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


# sin 커브
x = np.arange(0, 10, 0.1) # 0~10 까지 0.1 간격의 숫자 배열
y = np.sin(x)

plt.plot(x,y)


# ## 한 번에 두 개 그래프 그리기

# In[37]:


y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('sin and cos')
plt.legend(['sin', 'cos'])

plt.show()


# ## Subplot

# In[38]:


plt.subplot(2, 1, 1) # (2,1) 형태 플랏의 첫 번째 자리에 그리겠다
plt.plot(x, y_sin)
plt.title('sin')

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('cos')

plt.show()

