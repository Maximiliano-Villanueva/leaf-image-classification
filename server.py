# -*- coding: utf-8 -*-
"""
https://blog.keras.io/
"""

from PIL import Image
import numpy as np
import flask

import os

import pickle
import matplotlib.pyplot as plt

import cv2


from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def loadModel():
    
    global model
    
    model = pickle.load(open(os.path.join('benchmark', 'tree', 'resnet50', 'model2.pkl'), 'rb'))


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
    
    


@app.route("/predict", methods=["POST"])
def predict():
    
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}
    
    
    
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            
            """
            prepare image
            """
            #image = Image.open(flask.request.files.get("image").stream)
            
            pred = Predictor()
            
            pred.loadModel(name = 'resnet50', output_layer = -2)
            
            ima = np.asarray(Image.open(flask.request.files.get("image").stream).convert('RGB'). resize((224,224)))
            ima = ima / 255
            ima = np.array(ima)
            #ima = ima.reshape(ima.shape[0], -1)
            
            predicted = pred.simplePred(img = ima)[0]
            
            
            """
            image.save('test.jpg', 'JPEG')
            
            #image = image.resize((224,224))
            
            image.save('test2.jpg', 'JPEG')
            
            image = np.asarray(image)
            
            image = image.astype('float32')
            
            image[:,:,0] /= 255
            image[:,:,1] /= 255
            image[:,:,2] /= 255
            
            image = image /255
            
            #image = Image.fromarray(image)
                        
            image = cv2.imread('test2.jpg')
            
            image /= 255
            
            print(image)
        
            image = np.expand_dims(image, axis=0)
            """
            
            # classify the input image and then initialize the list
            # of predictions to return to the client
            #preds = model.predict_one(image)
            
            preds = model.predict_proba_one(predicted)
            #preds = model.predict_proba_one(image)
            
            data["maxClass"] = preds#int(np.argmax(preds))
                
            # indicate that the request was a success
            data["success"] = True
            
    # return the data dictionary as a JSON response
    return flask.jsonify(data)




# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server...", "please wait until server has fully started"))
    loadModel()
    app.run()