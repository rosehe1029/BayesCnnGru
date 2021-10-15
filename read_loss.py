import re
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
loss=[]

i=0
with open("model0Log.log", "r") as f:
    for line in f.readlines():
        i=i+1
        if i%2==0:
            continue
        line = line.strip('\n')
        print(line)
        e,l,ll,l1,l2,l3,ls=re.findall(r"[-+]?\d*\.\d+|\d+", line)
        print(l,l1,ls)
        loss.append(float(l))



print('loss',len(loss))
import matplotlib.pyplot as plt
import torch
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

x=np.arange(1,501,1)

fig = plt.figure(figsize = (7,5))
ax1 = fig.add_subplot(1, 1, 1)

pl.plot(x,loss,'b-',label=u'loss')#'r-'

pl.legend()

pl.legend()
pl.xlabel(u'epoch')
pl.ylabel(u'loss')
plt.title('loss for CNN-GRU in training')
plt.show()