# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: spectral_extraction.py extraction of the spectral signature, indices or texture features
             in a hdf5 table format for analysis and for the pixel-level semantic segmentation with 
             random forest classifier.
'''

import os
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from osgeo import gdal
from os.path import dirname as up
from assets import s2_mapping, mados_cat_mapping, conf_mapping

rev_cat_mapping = {v:k for k,v in mados_cat_mapping.items()}
rev_conf_mapping = {v:k for k,v in conf_mapping.items()}

def ImageToDataframe(RefImage, cols_mapping = {}, keep_annotated = True, prefix = '_rhorc_'):
    # This function transform an image with the associated class and 
    # confidence tif files (_cl.tif and _conf.tif) to a dataframe

    # Read patch
    ds = gdal.Open(RefImage)
    IM = np.copy(ds.ReadAsArray())

    # Read associated class patch
    ds_cl = gdal.Open(RefImage.replace(prefix , '_cl_'))
    IM_cl = np.copy(ds_cl.ReadAsArray())[np.newaxis, :, :]

    # Read associated confidence level patch
    ds_conf = gdal.Open(RefImage.replace(prefix , '_conf_'))
    IM_conf = np.copy(ds_conf.ReadAsArray())[np.newaxis, :, :]
    
    # Read associated class patch
    ds_rep = gdal.Open(RefImage.replace(prefix , '_rep_'))
    IM_rep = np.copy(ds_rep.ReadAsArray())[np.newaxis, :, :]

    # Stack all these together
    IM_T = np.moveaxis(np.concatenate([IM, IM_cl, IM_conf, IM_rep], axis = 0), 0, -1)
        
    bands = IM_T.shape[-1]
    IM_VECT = IM_T.reshape([-1,bands])

    IM_VECT = IM_VECT[IM_VECT[:,-3] > 0] # Keep only based on non zero class
        
    if cols_mapping:
        IM_df = pd.DataFrame({k:IM_VECT[:,v] for k, v in cols_mapping.items()})
    else:
        IM_df = pd.DataFrame(IM_VECT)

    ds = None
    ds_conf = None
    ds_cl = None
    ds_rep = None
    
    return IM_df

def main(options):

    mapping = s2_mapping
    h5_prefix = 'dataset'
    prefix = '_rhorc_'
        
    # Get patches files without _cl and _conf associated files
    patches = glob(os.path.join(options['path'],'*','*.tif'))
    
    patches = [p for p in patches if ('_cl_' not in p) and ('_conf_' not in p) and ('_rep_' not in p)]
    
    root_path = os.path.dirname(options['path'])

    # Read splits
    X_train = np.genfromtxt(os.path.join(options['path'], 'splits','train_X.txt'),dtype='str')
    
    X_val = np.genfromtxt(os.path.join(options['path'], 'splits','val_X.txt'),dtype='str')
    
    X_test = np.genfromtxt(os.path.join(options['path'], 'splits','test_X.txt'),dtype='str')
    
    dataset_name = os.path.join(root_path, h5_prefix + '_nonindex.h5')
    hdf = pd.HDFStore(dataset_name, mode = 'w')
    
    # For each patch extract the spectral signatures and store them
    for im_name in tqdm(patches):

        # Get date_tile_image info

        splited_name = os.path.basename(im_name).split('.tif')[0].split('_')
        img_name = '_'.join(splited_name[:-3]) + '_' + splited_name[-1]

        # Generate Dataframe from Image
        if img_name in X_train:
            split = 'Train'
            temp = ImageToDataframe(im_name, mapping, prefix = prefix)
        elif img_name in X_val:
            split = 'Validation'
            temp = ImageToDataframe(im_name, mapping, prefix = prefix)
        elif img_name in X_test:
            split = 'Test'
            temp = ImageToDataframe(im_name, mapping, prefix = prefix)
        else:
            raise AssertionError("Image not in train,val,test splits")
        
        temp['Scene'] = os.path.basename(im_name).split('_')[1]
        temp['Crop'] = os.path.basename(im_name).split('_')[-1].replace('.tif','')
        
        # Store data
        hdf.append(split, temp, format='table', data_columns=True, min_itemsize={'Class':27,
                                                                                 'Confidence':8,
                                                                                 'Crop':3,
                                                                                 'Scene':5})
    
    hdf.close()
    
    # Read the stored file and fix an indexing problem (indexes were not incremental and unique)
    hdf_old = pd.HDFStore(dataset_name, mode = 'r')
    
    df_train = hdf_old['Train'].copy(deep=True)
    df_val = hdf_old['Validation'].copy(deep=True)
    df_test = hdf_old['Test'].copy(deep=True)
    
    df_train.reset_index(drop = True, inplace = True)
    df_val.reset_index(drop = True, inplace = True)
    df_test.reset_index(drop = True, inplace = True)

    hdf_old.close()
    
    # Store the fixed table to a new dataset file
    dataset_name_fixed = os.path.join(root_path, h5_prefix+'.h5')

    df_train.to_hdf(dataset_name_fixed, key='Train', mode='a', format='table', data_columns=True)
    df_val.to_hdf(dataset_name_fixed, key='Validation', mode='a', format='table', data_columns=True)
    df_test.to_hdf(dataset_name_fixed, key='Test', mode='a', format='table', data_columns=True)
    
    os.remove(dataset_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--path', default='', help='Path to Images')

    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict
    
    main(options)
