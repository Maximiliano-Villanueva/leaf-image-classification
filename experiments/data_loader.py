# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from PIL import Image

import os

class DataLoader:
    
    def __init__(self, data_path, separator = None):
        
        self.data_path = data_path
        self.separator = separator
        
    
    def loadDatasetLab(self, path_column, label_column, new_encoded_column, drop_columns, kvfilter = None, path_data_join = ''):
        
        self._dataset = pd.read_csv(self.data_path) if self.separator is None else pd.read_csv(self.data_path, sep=self.separator)

                
        self._dataset = self._dataset[self._dataset['source'] == 'lab']
        
        
        if drop_columns is not None and len(drop_columns) > 0 :
            self._dataset.drop(drop_columns, axis= 1, inplace=True)
        
        self._dataset[path_column] = self._dataset[path_column].apply(lambda x: os.path.join(path_data_join, str(x).replace('/', '\\')))
        
        self._dataset[new_encoded_column] = self._dataset[label_column]
        
        self._dataset[new_encoded_column] = self._dataset[new_encoded_column].map(dict(zip(self._dataset[new_encoded_column].unique(), range(self._dataset[new_encoded_column].nunique()))))
        
        self._dataset[new_encoded_column] = self._dataset[new_encoded_column]
    
    def loadDataset(self, path_column, label_column, new_encoded_column, drop_columns, kvfilter = None, path_data_join = ''):
 
        self._dataset = pd.read_csv(self.data_path) if self.separator is None else pd.read_csv(self.data_path, sep=self.separator)

        if kvfilter is not None:
            
            for k in kvfilter:
                
                if isinstance(kvfilter[k], list):
                    self._dataset = self._dataset[self._dataset[k].isin(kvfilter[k])]
                else:
                    self._dataset = self._dataset[self._dataset[k] == kvfilter[k]]
                
                self._dataset = self._dataset[self._dataset['source'] == 'field']
        
        if drop_columns is not None and len(drop_columns) > 0 :
            self._dataset.drop(drop_columns, axis= 1, inplace=True)
        
        self._dataset[path_column] = self._dataset[path_column].apply(lambda x: os.path.join(path_data_join, str(x).replace('/', '\\')))
        
        self._dataset[new_encoded_column] = self._dataset[label_column]
        
        self._dataset[new_encoded_column] = self._dataset[new_encoded_column].map(dict(zip(self._dataset[new_encoded_column].unique(), range(self._dataset[new_encoded_column].nunique()))))
        
        self._dataset[new_encoded_column] = self._dataset[new_encoded_column]

    

    def getLabels(self, label_column):
                
        classMap = dict(zip(self._dataset[label_column].unique(), range(self._dataset[label_column].nunique())))
        
        inv_map = {v: k for k, v in classMap.items()}
        
        
        return inv_map
        
    def getNClasses(self, column):
        
        return self._dataset[column].nunique()
    
    
    
    def getTsneIndices(self, group_label, col):
        
        indices = self._dataset.groupby(group_label).count()[col].values
        
        for i in range(1,len(indices)):
            
            indices[i] += indices[i-1]
            
        return indices
    
    
    def getImageList(self, nClasses, path_col, reshape = True, width = 224, height = 224):
        
        images = list()
         
        #load images
        for j in range(0,nClasses):
        
            for index, row in self._dataset.iterrows():
                
                im_path = row[path_col]
                ima = np.asarray(Image.open(im_path).convert('RGB'). resize((width,height)))
                ima = ima / 255
            
                images.append(ima)
        
        if reshape:
                
            images = np.array(images)
            images = images.reshape(images.shape[0], -1)
            
        return images
            
        