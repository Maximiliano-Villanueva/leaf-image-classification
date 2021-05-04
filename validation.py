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

model_name = 'resnet50'#'resnet50'

dataset = stream.iter_csv(model_name + '.csv', target="class", converters=types)
dataset = stream.shuffle(dataset, buffer_size=50)



# construct our pipeline

"""
model = StandardScaler() | HoeffdingTreeClassifier(
    grace_period=100, split_confidence=1e-5)


model = StandardScaler() | OneVsRestClassifier(classifier=LogisticRegression())
"""

filename = 'model.pkl'

base_path = os.path.join('benchmark', 'tree', model_name)

model = HoeffdingTreeClassifier()
#model = pickle.load(open(os.path.join('benchmark', 'tree', 'resnet50', 'model2.pkl'), 'rb'))

metric = Accuracy()


acc_history = list()


start = time.time()

#for j in range(0,5):
for (i, (X, y)) in enumerate(dataset):
    
    preds = model.predict_one(X)
    model = model.learn_one(X, y)
    
    metric = metric.update(y, preds)
    
    print("INFO] update {} - {}".format(i, metric))
    
    acc_history.append(metric.get())


end = time.time()

print('elapsed time: ', end - start)

print("[INFO] final - {}".format(metric))


acc_history = np.array(acc_history)

base_path = os.path.join('benchmark', 'tree', model_name)

if not os.path.exists(base_path):
    os.makedirs(base_path)
    
if not os.path.exists(os.path.join(base_path, 'metrics')):
    os.makedirs(os.path.join(base_path, 'metrics'))

metrics_path = os.path.join(base_path, 'metrics', 'accuracy')


np.save(metrics_path, acc_history)

filename = os.path.join(base_path, 'model2.pkl')
pickle.dump(model, open(filename, 'wb'))
