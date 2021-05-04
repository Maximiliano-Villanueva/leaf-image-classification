# -*- coding: utf-8 -*-


from experiments.data_loader import DataLoader

import os

import sys

class ExperimentGenerator:
    
    
    
    @staticmethod
    def getDataLab(load_images = False):  

        path = os.path.join('..', 'mrcnn' ,'data', 'leafsnap-dataset-images.txt')
        
        width = 224
        height = 224
        
        dl = DataLoader(data_path=path, separator='\t')
        
        #path_column, label_column, new_encoded_column, drop_columns, kvfilter = None, path_data_join = ''
        dl.loadDatasetLab(path_column = 'image_path',
                                label_column = 'species',
                                new_encoded_column = 'label2',
                                drop_columns = ['segmented_path', 'source', 'file_id'],
                                kvfilter = {'source' : 'lab'},
                                path_data_join = 'data')
        
        
        
        
        
        
        dl._dataset = dl._dataset[(dl._dataset['label2'] > 1) & (dl._dataset['label2'] <7)]
        
        dl._dataset = dl._dataset.groupby('label2').head(15).reset_index(drop = True)
        #dl._dataset = dl._dataset.groupby('label2')[:15]
        
        dl._dataset['label2'] = dl._dataset['label2'].apply(lambda x : int(x)+1)
        
        nClasses = dl.getNClasses(column = 'label2')
        
        dl._dataset['image_path'] = dl._dataset['image_path'].apply(lambda x : x.replace(os.path.join('data','dataset'), os.path.join('..', 'mrcnn' ,'data', 'dataset')))
        dl._dataset['image_path'] = dl._dataset['image_path'].apply(lambda x : x.replace('field', 'rotate_lab'))
        #dl._dataset['image_path'] = dl._dataset['image_path'].apply(lambda x : x.replace('field', 'segmentated_lab'))
        dl._dataset['mask_path_seg'] = dl._dataset['image_path'].apply(lambda x : x.replace('field', 'segmentated'))
        
        #dl._dataset['label2'] = dl._dataset['label2'].apply(lambda x : int(x) - 3)

        if load_images:
            indices = dl.getTsneIndices(group_label = 'label2', col = 'image_path')

            #images = dl.getImageList(nClasses = nClasses, path_col = 'image_path', reshape = True, width = width, height = height)
            
            return dl, nClasses, indices#, images
        
        return dl, nClasses 
        
        
        #return dl
        
    
    
    @staticmethod
    def getData(load_images = False):  

        path = os.path.join('..', 'mrcnn' ,'data', 'leafsnap-dataset-images.txt')
        
        width = 224
        height = 224
        
        dl = DataLoader(data_path=path, separator='\t')
        
        #path_column, label_column, new_encoded_column, drop_columns, kvfilter = None, path_data_join = ''
        dl.loadDataset(path_column = 'image_path',
                                label_column = 'species',
                                new_encoded_column = 'label2',
                                drop_columns = ['segmented_path', 'source', 'file_id'],
                                kvfilter = {'source' : 'field'},
                                path_data_join = 'data')
        
        
        
        
        
        
        dl._dataset = dl._dataset[(dl._dataset['label2'] > 1) & (dl._dataset['label2'] <7)]
        
        dl._dataset = dl._dataset.groupby('label2').head(15).reset_index(drop = True)
        #dl._dataset = dl._dataset.groupby('label2')[:15]
        
        dl._dataset['label2'] = dl._dataset['label2'].apply(lambda x : int(x)+1)
        
        nClasses = dl.getNClasses(column = 'label2')
        
        dl._dataset['image_path'] = dl._dataset['image_path'].apply(lambda x : x.replace(os.path.join('data','dataset'), os.path.join('..', 'mrcnn' ,'data', 'dataset')))
        dl._dataset['image_path'] = dl._dataset['image_path'].apply(lambda x : x.replace('field', 'rotate'))
        #dl._dataset['image_path'] = dl._dataset['image_path'].apply(lambda x : x.replace('field', 'segmentated'))
        dl._dataset['mask_path_seg'] = dl._dataset['image_path'].apply(lambda x : x.replace('field', 'segmentated'))
        
        #dl._dataset['label2'] = dl._dataset['label2'].apply(lambda x : int(x) - 3)

        if load_images:
            indices = dl.getTsneIndices(group_label = 'label2', col = 'image_path')

            #images = dl.getImageList(nClasses = nClasses, path_col = 'image_path', reshape = True, width = width, height = height)
            
            return dl, nClasses, indices#, images
        
        return dl, nClasses 
        
        
        #return dl
        
        
    @staticmethod
    def getTestData(load_images = False):  

        path = os.path.join('..', 'mrcnn' ,'data', 'leafsnap-dataset-images.txt')
        
        width = 224
        height = 224
        
        dl = DataLoader(data_path=path, separator='\t')
        
        #path_column, label_column, new_encoded_column, drop_columns, kvfilter = None, path_data_join = ''
        dl.loadDataset(path_column = 'image_path',
                                label_column = 'species',
                                new_encoded_column = 'label2',
                                drop_columns = ['segmented_path', 'source', 'file_id'],
                                kvfilter = {'source' : 'field'},
                                path_data_join = 'data')
        
        
        
        
        
        
        dl._dataset = dl._dataset[(dl._dataset['label2'] > 1) & (dl._dataset['label2'] <7)]
        
        dl._dataset = dl._dataset.groupby('label2').tail(25).reset_index(drop = True)
        #dl._dataset = dl._dataset.groupby('label2')[15:]
        
        dl._dataset['label2'] = dl._dataset['label2'].apply(lambda x : int(x)+1)
        
        nClasses = dl.getNClasses(column = 'label2')
        
        dl._dataset['image_path'] = dl._dataset['image_path'].apply(lambda x : x.replace(os.path.join('data','dataset'), os.path.join('..', 'mrcnn' ,'data', 'dataset')))
        dl._dataset['image_path'] = dl._dataset['image_path'].apply(lambda x : x.replace('field', 'rotate'))
        #dl._dataset['image_path'] = dl._dataset['image_path'].apply(lambda x : x.replace('field', 'segmentated'))
        dl._dataset['mask_path_seg'] = dl._dataset['image_path'].apply(lambda x : x.replace('field', 'segmentated'))
        
        
        if load_images:
            indices = dl.getTsneIndices(group_label = 'label2', col = 'image_path')

            #images = dl.getImageList(nClasses = nClasses, path_col = 'image_path', reshape = True, width = width, height = height)
            
            return dl, nClasses, indices#, images
        
        return dl, nClasses 
    
    
    @staticmethod
    def getTestDataLab(load_images = False):  

        path = os.path.join('..', 'mrcnn' ,'data', 'leafsnap-dataset-images.txt')
        
        width = 224
        height = 224
        
        dl = DataLoader(data_path=path, separator='\t')
        
        #path_column, label_column, new_encoded_column, drop_columns, kvfilter = None, path_data_join = ''
        dl.loadDataset(path_column = 'image_path',
                                label_column = 'species',
                                new_encoded_column = 'label2',
                                drop_columns = ['segmented_path', 'source', 'file_id'],
                                kvfilter = {'source' : 'lab'},
                                path_data_join = 'data')
        
        
        return dl, ''
        
        
        
        dl._dataset = dl._dataset[(dl._dataset['label2'] > 1) & (dl._dataset['label2'] <7)]
        
        dl._dataset = dl._dataset.groupby('label2').tail(15).reset_index(drop = True)
        #dl._dataset = dl._dataset.groupby('label2')[:15]
        
        dl._dataset['label2'] = dl._dataset['label2'].apply(lambda x : int(x)+1)
        
        nClasses = dl.getNClasses(column = 'label2')
        
        dl._dataset['image_path'] = dl._dataset['image_path'].apply(lambda x : x.replace(os.path.join('data','dataset'), os.path.join('..', 'mrcnn' ,'data', 'dataset')))
        dl._dataset['image_path'] = dl._dataset['image_path'].apply(lambda x : x.replace('field', 'rotate'))
        dl._dataset['mask_path_seg'] = dl._dataset['image_path'].apply(lambda x : x.replace('field', 'segmentated'))
        
        #dl._dataset['label2'] = dl._dataset['label2'].apply(lambda x : int(x) - 3)

        if load_images:
            indices = dl.getTsneIndices(group_label = 'label2', col = 'image_path')

            #images = dl.getImageList(nClasses = nClasses, path_col = 'image_path', reshape = True, width = width, height = height)
            
            return dl, nClasses, indices#, images
        
        return dl, nClasses 
        
        
        #return dl
        
