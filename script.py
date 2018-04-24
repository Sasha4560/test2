import numpy as np
import scipy.io
import matplotlib.pyplot as plt

#1 
a = 10
print(a)
print()
a = [1, 2, 3]
print(a)
print()
a = np.array([1, 2, 3])
print(a)
print()
a = [[1], [1, 2], [1, 2, 3]]
print(a)
print()
a = np.zeros((2, 3))
print(a)
print()
a.shape[0]	# 2
a.shape[1]	# 3
print(a)
print()
b = a.reshape((1, 6))
print(b)
print()
b = a.reshape((1, -1))
print(b)
print()
a = np.ones((2, 3)) * 7
print(a)
print()
a = np.random.random((2, 3))
print(a)
print()
a = 5 * np.random.random((2, 3)) + 10
print(a)
print()
a = np.random.randint(2, 6, (5, 5))
print(a)
print()
data = np.random.normal(3, 5, 10)
print(data)
print()

#2 
data = np.loadtxt('D:\\data.txt')
print(data)
print()
data = np.loadtxt('D:\\data.txt', dtype=np.int32)
print(data)
print()
import os
os.chdir('D:')
os.getcwd 

#3
data = scipy.io.loadmat('D:/data/1D/var6.mat')
n = data['p']
max_    = np.max(n)
min_    = np.min(n)
mean_   = np.mean(n)
median_ = np.median(n)
var_    = np.var(n)
std_    = np.std(n) 

#4

plt.plot(n)
plt.hlines(mean_, 0, len(n), colors='r', linestyles='solid')
plt.hlines(mean_ + std_, 0, len(n), colors='g', linestyles='dashed')
plt.hlines(mean_ - std_, 0, len(n), colors='g', linestyles='dashed')
plt.show()

plt.hist(n, bins=20)
plt.grid()
plt.show()

#5

def autocorrelate(a):
  n = len(a)
  cor = []
  for i in range(n//2, n//2+n):
    a1 = a[:i+1]   if i < n else a[i-n+1:]
    a2 = a[n-i-1:] if i < n else a[:2*n-i-1]
    cor.append(np.corrcoef(a1, a2)[0, 1])
  return np.array(cor)

n_1d = np.ravel(n)
cor = autocorrelate(n_1d)
plt.plot(cor)
plt.show()

#6

data = scipy.io.loadmat('D:/data/ND/var1.mat')
mn = data['mn']

#7
n = mn.shape[1]
corr_matrix = np.zeros((n, n))

for i in range(0, n):
  for j in range(0, n):
    corr_matrix[i, j] = np.corrcoef(mn[:, i], mn[:, j])[0, 1]

np.set_printoptions(precision=2)
print(corr_matrix)

plt.plot(mn[:, 2], mn[:, 5], 'b.')
plt.grid()
plt.show()