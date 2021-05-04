"""
predict classes
"""
#dataset['prediction'] = dataset['id'].apply(lambda x : predictClass(model, os.path.join('data', 'images', x), width, height))

from river.preprocessing import StandardScaler

from river.metrics import Accuracy
from river import stream
from river import compat
from river.tree import HoeffdingTreeClassifier
import numpy as np


from river.linear_model import LogisticRegression
from river.multiclass import OneVsRestClassifier

import time
import os

import pickle


"""
filename = 'model.pkl'
#pickle.dump(model, open(filename, 'wb'))

model = pickle.load(open(filename, 'rb'))
"""

types = {"feat_{}".format(i): float for i in range(0, 7 * 7 * 2048)}
types["class"] = int

model_name = 'resnet50_test'#'resnet50'

dataset = stream.iter_csv(model_name + '.csv', target="class", converters=types)
#dataset = stream.shuffle(dataset, buffer_size=50)


filename = 'model.pkl'

base_path = os.path.join('benchmark', 'tree', model_name)

model = pickle.load(open(os.path.join('benchmark', 'tree', 'resnet50', 'model2.pkl'), 'rb'))

metric = Accuracy()


acc_history = list()


start = time.time()

#for j in range(0,5):
for (i, (X, y)) in enumerate(dataset):
    
    preds = model.predict_one(X)
    #print(model.predict_proba_one(X))
    
    metric = metric.update(y, preds)
    
    print("INFO] update {} - {}".format(i, metric))
    print((y, preds))
    
    acc_history.append(metric.get())


end = time.time()

print('elapsed time: ', end - start)

print("[INFO] final - {}".format(metric))


acc_history = np.array(acc_history)





