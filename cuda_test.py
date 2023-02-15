import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt

sizelist = [500, 1000, 5000, 10000, 20000]
#cut = np.zeros(np.size(sizelist))
#npt = np.zeros(np.size(sizelist))

# for i in range(np.size(sizelist)):
#     startloop = time.time()
#     cua = cp.random.randint(0,50,(sizelist[i],sizelist[i]))
#     cub = cp.random.randint(0,50,(sizelist[i],sizelist[i]))
#     start = time.time()
#     c = cua*cub
#     end = time.time()
#     cut[i] = end-start

#     npa = np.random.randint(0,50,(sizelist[i],sizelist[i]))
#     npb = np.random.randint(0,50,(sizelist[i],sizelist[i]))
#     start = time.time()
#     c = npa*npb
#     end = time.time()
#     npt[i] = end-start
#     endloop = time.time()
#     print(f'{sizelist[i]} size took {endloop-startloop}s')
# plt.plot(sizelist, cut, label='cupy 1x')
# plt.plot(sizelist, npt, label='numpy 1x')
# plt.legend()
# plt.show()

size = 5000
powerlist = np.asarray([6,8,10,20])
cut = np.zeros(np.size(powerlist))
npt = np.zeros(np.size(powerlist))
for i in range(np.size(powerlist)):
    startloop = time.time()
    start = time.time()
    c = 1
    for j in range(powerlist[i]):
        c *= cp.random.randint(0,60,(size, size)) 
    end = time.time()
    cut[i] = end-start

    start = time.time()
    c = 1
    for j in range(powerlist[i]):
        c *= np.random.randint(0, 60, (size,size))
    end = time.time()
    npt[i] = end-start
    endloop = time.time()
    print(f'{powerlist[i]} size took {endloop-startloop}s')
plt.plot(powerlist, cut, label='cupy 1x')
plt.plot(powerlist, npt, label='numpy 1x')
plt.xlabel('order of multiplication')
plt.ylabel('execution time (s)')
plt.legend()
plt.show()
print("Over")