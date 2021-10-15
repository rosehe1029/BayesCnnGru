from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, recall_score, roc_curve, roc_auc_score
from sklearn.metrics.ranking import auc

import pandas  as pd
import matplotlib.pyplot as plt
test_label=pd.read_csv("111test_labels0.csv", header=None)
print(test_label.shape)

p=pd.read_csv("pred.csv", header=None)
p=p.iloc[:,0]
print(p.shape)
pp=[i>0.5 for i in p]
fpr, tpr, threshold = roc_curve(test_label, p)
roc_auc = auc(fpr, tpr)
print("MCC: %f " %matthews_corrcoef(test_label,pp))
print("AUC",roc_auc)
#temp = confusion_matrix(test_label, p)
#TN, FP, FN, TP = temp.ravel()
acc= accuracy_score(test_label, pp)#(TP + TN) / (TN + FP + FN + TP)
print(acc)
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='red',lw=2, label='Proposed Method :ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=22)
plt.ylabel('True Positive Rate',fontsize=22)
plt.title('AUC:ROC curve',fontsize=22)

'''
plt.title('ROC curve comparison of the models',fontsize=22)
#textCNN
Ctest_label=pd.read_csv("347.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("347_.csv", header=None)
Cp=Cp.iloc[:,0]
print(Cp.shape)
pp=[i>0.5 for i in Cp]
Cp=Cp
fpr, tpr, threshold = roc_curve(Ctest_label, Cp)
roc_auc = auc(fpr, tpr)
print("MCC: %f " %matthews_corrcoef(Ctest_label,pp))
print(roc_auc)
acc= accuracy_score(Ctest_label, pp)#(TP + TN) / (TN + FP + FN + TP)
print("acc",acc)
plt.plot(fpr, tpr, color='blue',lw=2, label='LSTM:ROC curve (area = %0.2f)' % roc_auc) 


#Data  A
Ctest_label=pd.read_csv("3.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("result/3.csv", header=None)
Cp=Cp.iloc[:,0]
print(Cp.shape)
pp=[i>0.5 for i in Cp]
fpr, tpr, threshold = roc_curve(Ctest_label, pp)
roc_auc = auc(fpr, tpr)
print("MCC: %f " %matthews_corrcoef(test_label,Cp))
print(roc_auc)
acc= accuracy_score(Ctest_label, pp)#(TP + TN) / (TN + FP + FN + TP)
print(acc)
plt.plot(fpr, tpr, color='green',lw=2, label='RF:ROC curve (area = %0.2f)' % roc_auc)


Ctest_label=pd.read_csv("310.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("result/310.csv", header=None)
Cp=Cp.iloc[:,0]
print(Cp.shape)
pp=[i>0.5 for i in Cp]
Cp=Cp
fpr, tpr, threshold = roc_curve(Ctest_label, Cp)
roc_auc = auc(fpr, tpr)
print("MCC: %f " %matthews_corrcoef(Ctest_label,pp))
print(roc_auc)
acc= accuracy_score(Ctest_label, pp)#(TP + TN) / (TN + FP + FN + TP)
print("acc",acc)
plt.plot(fpr, tpr, color='orange',lw=2, label='SVM:ROC curve (area = %0.2f)' % roc_auc) 

#RF
Ctest_label=pd.read_csv("4.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("result/4.csv", header=None)
Cp=Cp.iloc[:,0]
print(Cp.shape)
pp=[i>0.5 for i in Cp]
Cp=Cp
fpr, tpr, threshold = roc_curve(Ctest_label, Cp)
roc_auc = auc(fpr, tpr)
print("MCC: %f " %matthews_corrcoef(Ctest_label,pp))
print(roc_auc)
acc= accuracy_score(Ctest_label, pp)#(TP + TN) / (TN + FP + FN + TP)
print("acc",acc)
plt.plot(fpr, tpr, color='yellow',lw=2, label='DNN:ROC curve (area = %0.2f)' % roc_auc) 

#plt.plot(fpr, tpr, color='yellow',lw=2, label='LR:ROC curve (area = %0.2f)' % roc_auc) 
#lstm
Ctest_label=pd.read_csv("28.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("result/28.csv", header=None)
Cp=Cp.iloc[:,0]
print(Cp.shape)
pp=[i>0.5 for i in Cp]
Cp=Cp
fpr, tpr, threshold = roc_curve(Ctest_label, Cp)
roc_auc = auc(fpr, tpr)
print("MCC: %f " %matthews_corrcoef(Ctest_label,pp))
print(roc_auc)
acc= accuracy_score(Ctest_label, pp)#(TP + TN) / (TN + FP + FN + TP)
print("acc",acc)
plt.plot(fpr, tpr, color='purple',lw=2, label='LR:ROC curve(area = %0.2f)' % roc_auc ) 



#DNN
Ctest_label=pd.read_csv("result/179.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("179.csv", header=None)
Cp=Cp.iloc[:,0]
print(Cp.shape)
pp=[i>0.5 for i in Cp]
Cp=Cp
fpr, tpr, threshold = roc_curve(Ctest_label, Cp)
roc_auc = auc(fpr, tpr)
print("MCC: %f " %matthews_corrcoef(Ctest_label,pp))
print(roc_auc)
acc= accuracy_score(Ctest_label, pp)#(TP + TN) / (TN + FP + FN + TP)
print("acc",acc)
plt.plot(fpr, tpr, color='pink',lw=2, label='KNN:ROC curve (area = %0.2f)' % roc_auc) 
#LR
'''
plt.legend(loc=0)
plt.legend(loc="lower right")
plt.show()
