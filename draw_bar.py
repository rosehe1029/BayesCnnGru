# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


name_list = ['-CNN-', '-GRU-','CNN-GRU']
num_list = [2,0,2]
num_list1 = [0,3,4]
num_list2 = [0.82,0.86,0.92]
num_list3 = [0.24,0.31,0.63]
num_list4 = [0.80,0.83,0.88]
print(len(num_list))
l=len(num_list)+3
x = list(range(len(num_list)))
total_width, n = 1, 5
width = total_width / n

plt.bar(x, num_list, width=width, label='-CNN- layer numbers', fc='mistyrose')


for i in range(len(x)):
    x[i] = x[i] + width

plt.bar(x, num_list1, width=width, label='-GRU- layer numbers', tick_label=name_list, fc='cyan')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label='AUC', tick_label=name_list, fc='yellow')


for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list3, width=width, label='MCC', tick_label=name_list, fc='palegreen')


for i in range(len(x)):
    x[i] = x[i] + width
    

plt.bar(x, num_list4, width=width, label='ACC', tick_label=name_list, fc='violet')
'''
name_list = ['KNN', 'LR', 'DNN','SVM','RF','LSTM','CNN-GRU']
num_list = [0.61,0.64,0.73,0.76,0.81,0.85,0.92]
num_list1 = [0.33,0.21,0.59,0.43,0.27,0.49,0.63]
num_list2 = [0.80,0.77,0.82,0.86,0.80,0.81,0.88]

print(len(num_list))
l = len(num_list) + 3
x = list(range(len(num_list)))
total_width, n = 2, 7
width = total_width / n

plt.bar(x, num_list, width=width, label='AUC', fc='violet')

for i in range(len(x)):
    x[i] = x[i] + width

plt.bar(x, num_list1, width=width, label='MCC', tick_label=name_list, fc='cyan')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label='ACC', tick_label=name_list, fc='ORANGE')

plt.title(u"AUC,MCC,ACC comparisons across different models")
plt.legend(loc='upper left')
#plt.legend()



name_list = ['without Data Augmentation', 'with Data Augmentation']
num_list = [0.88,0.92]
num_list1 = [0.49,0.63]
num_list2 = [0.82,0.88]
x = list(range(len(num_list)))
total_width, n = 0.9, 3
width = total_width / n

plt.bar(x, num_list, width=width, label='AUC', fc='khaki')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='MCC', tick_label=name_list, fc='lightblue')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label='ACC', tick_label=name_list, fc='lightpink')
plt.title(u"with/without Data Augmentation")
'''
plt.title(u"Ablation Experiments")
plt.legend(loc='upper left')
plt.show()
