from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, recall_score, roc_curve, roc_auc_score
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
import torch
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def config():
    print('\n\tInitializing experiment configurations...')
    exp_settings = {}
    exp_settings['batch_size'] = 128
    exp_settings['num_epochs'] = 100
    exp_settings['patience'] = 10
    exp_settings['window_size'] = 20
    exp_settings['space'] = {'lr': hp.uniform('lr', 0.00001, 0.001), 'num_kernels': 1 + hp.randint('num_kernels', 2)}
    exp_settings['folder'] = 'experiment_output11111'
    if not os.path.exists(exp_settings['folder']): os.mkdir(exp_settings['folder'])
    exp_settings['datafile'] = exp_settings['folder'] + '/data_summary.txt'
    exp_settings['run_file'] = exp_settings['folder'] + '/eval_summary.txt'
    exp_settings['run_counter'] = 0
    exp_settings['glob_loss'] = 10.0
    return exp_settings



class ProtPep_dataset(Dataset):

    def __init__(self, ws, mode):

        self.ws = ws
        self.mode = mode

        input_file = self.mode + '_' + str(self.ws) + '_set.csv'
        label_file = str(self.mode) + '_labels.txt'

        self.image_list = []
        self.label_list = []

        with open(input_file, 'r') as input_f:

            buffer = csv.reader(input_f)
            img = []
            for i, row in enumerate(buffer):
                row = [float(n) for n in row]
                img.append(row)

                if (i + 1) % self.ws == 0:
                    self.image_list.append(np.array(img))
                    # print(len(img))
                    img.clear()

        with open(label_file, 'r') as label_f:
            self.label_list = list(label_f.read())
            # print(len(self.label_list))

    def shuffle_lists(self, l1, l2):
        random.seed(4)
        mapIndexPosition = list(zip(l1, l2))
        random.shuffle(mapIndexPosition)
        l1, l2 = zip(*mapIndexPosition)
        return list(l1), list(l2)

    def __getitem__(self, index):
        # plt.imshow(self.image_list[index], 'gray')
        # plt.savefig('image.png')
        # plt.show()
        #print(self.image_list)
        #print(len(self.image_list))
        #print(self.label_list)
        #print(len(self.label_list))
        return torch.tensor(self.image_list[index]), torch.tensor(int(self.label_list[index]))

    def __len__(self):
        return len(self.label_list)



def load_all_data(exp_settings):
    def load_data(mode):
        dataset = ProtPep_dataset(ws=exp_settings['window_size'], mode=mode)
        return DataLoader(dataset, shuffle=True, batch_size=exp_settings['batch_size'])

    def save_details():
        for mode, loader in sets.items():
            print('Summarizing ', mode, ' set...')
            with open(exp_settings['datafile'], 'a') as f:
                f.write('\n' + mode + ' set contains ' + str(len(loader.dataset)) + ' residues')

    print('\n\tLoading datasets...')
    sets = {}
    sets['test'] = load_data('test')
    save_details()
    return sets


import pickle
f = open('experiment_output/run111.pickle','rb')
s = f.read()
model = pickle.loads(s)
print(model)


import pandas as pd

#print(model)
data = pd.read_csv("test_20_set.csv",header=None)

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
exp_settings = config()
data = load_all_data(exp_settings)
iter_dict={}

def testcalc_metrics(iter_dict):
    scores = {}
    #print(iter_dict['val_labels'])
    #print(iter_dict['val_outputs'])
    temp = confusion_matrix(iter_dict['val_labels'], iter_dict['val_outputs'])
    TN, FP, FN, TP = temp.ravel()
    # scores['misclassification_rate'] = (FP + FN) / (TN + FP + FN + TP ) # or 1-accuracy
    scores['sensitivity'] = TP / (FN + TP)  # aka sensitivity or recall
    # scores['false_pos_rate'] = FP / (TN + FP)
    scores['specificity'] = TN / (TN + FP)  # aka specificity
    precision = TP / (TP + FP)
    # scores['prevalence'] = (TP + FN) / (TN + FP + FN + TP )
    scores['f_score'] = (2 * scores['sensitivity'] * precision) / (scores['sensitivity'] + precision)

    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if mcc_denominator == 0: mcc_denominator = 1
    scores['mcc'] = mcc_numerator / math.sqrt(mcc_denominator)
    scores['auc'] = roc_auc_score(iter_dict['val_labels'], iter_dict['output_prob'])
    scores['accuracy'] = (TP + TN) / (TN + FP + FN + TP)
    scores['conf_matrix'] = temp

    return scores

#ytest=model.test(data.values)
from sklearn.utils.multiclass import type_of_target

iter_dict['val_labels']=[]
iter_dict['val_outputs']=[]
val_outputs, val_labels,val_prob = [],  [], []  # for metrics calculations

for (images,labels) in data['test']:
    inputs = Variable(torch.tensor(images)).to(device)
    inputs = inputs.unsqueeze(1)
    labels = Variable(labels.long()).to(device)
    outputs = model(inputs)
    out_max = outputs.detach()
    val_prob.append(out_max)
    out_max = torch.argmax(out_max, dim=1)
    val_outputs.append(out_max)
    val_labels.append(labels)
    #print(type_of_target(labels.cpu().numpy()))
    #print(type_of_target(outputs.cpu().detach().numpy()))

iter_dict['val_outputs'] = torch.cat(val_outputs).cpu().numpy()
iter_dict['val_labels'] = torch.cat(val_labels).cpu().numpy()
iter_dict['output_prob'] = torch.cat(val_prob).cpu().numpy()[:, 1:]

print(testcalc_metrics(iter_dict))