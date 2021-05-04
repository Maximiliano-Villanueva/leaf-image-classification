# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
import numpy as np

class Predictor:
    
    def __init__(self):
        pass
    
    
    def loadModel(self, name, output_layer = None):
        
        model = None
        
        name = name.lower()
        
        self._model_name = name
    
        model = ResNet50V2(weights='imagenet',include_top=False)
        self._last_dim = 7
        model = Model(inputs = model.inputs, outputs = model.layers[output_layer].output)
        
        self._model = model
                
    
    
    def simplePred(self, img):
        
        img = np.expand_dims(img, axis=0)
        features = self._model.predict(img)
        #7 * 7 * 2048

        features = features.reshape((features.shape[0], 7 * 7 * 2048))
        #features = features.reshape((features.shape[0], self._last_dim * self._last_dim * self._model.layers[-1].output.shape[-1]))
        
        return features
    
    
    
    
    

from experiments.ExperimentGenerator import ExperimentGenerator
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt

width, height = 224, 224

df = None
nClasses = None

#dl, nClasses = ExperimentGenerator.getData()

#dl, nClasses = ExperimentGenerator.getDataLab()
    




dl, nClasses= ExperimentGenerator.getTestData()



pred = Predictor()

model_name = 'resnet50_test'

pred.loadModel(name = model_name, output_layer = -2)


        
predictedValues = list()

label_list = list()


for index, row in dl._dataset.iterrows():
    try:
        im_path = row['image_path']
        print(im_path)
        
        label_list.append(row['label2'])
        
        ima = np.asarray(Image.open(im_path).convert('RGB'). resize((width,height)))
        ima = ima / 255
        ima = np.array(ima)

        #ima = ima.reshape(ima.shape[0], -1)
        
        predicted = pred.simplePred(img = ima)[0]
        
        predictedValues.append(predicted)
    except:
        print('pass')
        pass

    
    
        
#predicted = pred.simplePred(img = images[image_index])[0]

#predictedValues.append(predicted)
    

    


#cols = ["feat_{}".format(i) for i in range(0, pred._last_dim * pred._last_dim * pred._model.layers[-1].output.shape[-1])]
cols = ["feat_{}".format(i) for i in range(0, 7 * 7 * 2048)]
cols = ["class"] + cols

csv = open(model_name + '.csv', "w")
csv.write("{}\n".format(",".join(cols)))

#for (label, vec) in zip(dataset.label2.values, predictedValues)  :
for (label, vec) in zip(label_list, predictedValues)  :

    vec = ",".join([str(v) for v in vec])
    csv.write("{},{}\n".format(label, vec))
    
csv.close()
