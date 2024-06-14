# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: evaluation.py includes the code in order to produce
             the evaluation for each class as well as the prediction
             masks for the pixel-level semantic segmentation.
'''

import os
import sys
import random
import logging
import rasterio
from rasterio.enums import Resampling
import argparse
from glob import glob
import numpy as np
from tqdm import tqdm
from os.path import dirname as up

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn import functional as F

sys.path.append(up(os.path.abspath(__file__)))
from marinext_wrapper import MariNext

sys.path.append(os.path.join(up(up(os.path.abspath(__file__))), 'utils'))
from dataset import MADOS, bands_mean, bands_std
from test_time_aug import TTA
from metrics import Evaluation, confusion_matrix
from assets import labels, bool_flag

root_path = up(up(os.path.abspath(__file__)))

logging.basicConfig(filename=os.path.join(root_path, 'logs','evaluating_marinext.log'), filemode='a',level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.info('*'*10)

def seed_all(seed):
    # Pytorch Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_band(path):
    return int(path.split('_')[-2])


def main(options):
    seed_all(0)
    # Transformations
    
    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(bands_mean, bands_std)
    
    splits_path = os.path.join(options['path'],'splits')
    
    # Construct Data loader

    dataset_test = MADOS(options['path'], splits_path, options['split'])

    test_loader = DataLoader(   dataset_test, 
                                batch_size = options['batch'], 
                                shuffle = False)

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    models_list = []
    
    models_files = glob(os.path.join(options['model_path'],'*.pth'))
    for model_file in models_files:
    
        model = MariNext(options['input_channels'], options['output_channels'])
    
        model.to(device)
    
        # Load model from specific epoch to continue the training or start the evaluation
        
        logging.info('Loading model files from folder: %s' % model_file)
    
        checkpoint = torch.load(model_file, map_location = device)
        checkpoint = {k.replace('decoder','decode_head'):v for k,v in checkpoint.items() if ('proj1' not in k) and ('proj2' not in k)}
    
        model.load_state_dict(checkpoint)

        del checkpoint  # dereference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        model.eval()
        
        models_list.append(model)

    y_true = []
    y_predicted = []
                             
    with torch.no_grad():
        for (image, target) in tqdm(test_loader, desc="testing"):

            if options['test_time_augmentations'] and options['batch']==1: # Only with batch = 1
                image = TTA(image)            

            image = image.to(device)
            target = target.to(device)
            
            seed_all(0)
            
            all_predictions = []
            for model in models_list:
                logits = model(image)
                logits = F.upsample(input=logits, size=(
                    target.shape[-2], target.shape[-1]), mode='bilinear')
                
                # Accuracy metrics only on annotated pixels
                probs = torch.nn.functional.softmax(logits, dim=1)
                predictions = probs.argmax(1)
                
                if options['test_time_augmentations'] and options['batch']==1: # Only with batch = 1
                    predictions = TTA(predictions, reverse_aggregation = True)
                
                all_predictions.append(predictions)
            all_predictions = torch.cat(all_predictions)
            all_predictions = torch.mode(all_predictions, dim=0, keepdim=True)[0]
                
            
            predictions = predictions.reshape(-1)
            target = target.reshape(-1)
            mask = target != -1
            
            predictions = predictions[mask].cpu().numpy()
            target = target[mask]
            
            target = target.cpu().numpy()
            
            y_predicted += predictions.tolist()
            y_true += target.tolist()
        
        ####################################################################
        # Save Scores to the .log file                                     #
        ####################################################################
        acc = Evaluation(y_predicted, y_true)
        logging.info("\n")
        logging.info("STATISTICS: \n")
        logging.info("Evaluation: " + str(acc))
        print("Evaluation: " + str(acc))
        conf_mat = confusion_matrix(y_true, y_predicted, labels, options['results_percentage'])
        logging.info("Confusion Matrix:  \n" + str(conf_mat.to_string()))
        print("Confusion Matrix:  \n" + str(conf_mat.to_string()))
        
                        
        seed_all(0)
                
        if options['predict_masks']:
            
            path = options['path']
            tiles = glob(os.path.join(path,'*'))
            ROIs_split = np.genfromtxt(os.path.join(splits_path, options['split']+'_X.txt'),dtype='str')

            impute_nan = np.tile(bands_mean, (240,240,1))

            for tile in tqdm(tiles, desc = 'testing'):

                # Get the number of different crops for the specific tile
                splits = [f.split('_cl_')[-1] for f in glob(os.path.join(tile, '10', '*_cl_*'))]
                
                for crop in splits:
                    crop_name = os.path.basename(tile)+'_'+crop.split('.tif')[0]
                    
                    if crop_name in ROIs_split:
        
                        # Load Input Images
                        # Get the bands for the specific crop 
                        all_bands = glob(os.path.join(tile, '*', '*L2R_rhorc*_'+crop))
                        all_bands = sorted(all_bands, key=get_band)
            
                        ################################
                        # Upsample the bands #
                        ################################
                        current_image = []
                        for c, band in enumerate(all_bands, 1):
                            upscale_factor = int(os.path.basename(os.path.dirname(band)))//10
            
                            with rasterio.open(band, mode ='r') as src:
                                tags = src.tags().copy()
                                meta = src.meta
                                dtype = src.read(1).dtype
                                current_image.append(src.read(1,
                                                                out_shape=(
                                                                    int(src.height * upscale_factor),
                                                                    int(src.width * upscale_factor)
                                                                ),
                                                                resampling=Resampling.nearest
                                                              ).copy()
                                                  )
                        
                        image = np.stack(current_image)
                        image = np.moveaxis(image, (0, 1, 2), (2, 0, 1))
            
                        os.makedirs(options['gen_masks_path'], exist_ok=True)
                    
                        output_image = os.path.join(options['gen_masks_path'], os.path.basename(crop_name).split('.tif')[0] + '_marinext.tif')
                    
                        # Update meta to reflect the number of layers
                        meta.update(count = 1)
                    
                        # Write it
                        with rasterio.open(output_image, 'w',
                                                    driver='GTiff',
                                                    height=image.shape[0],
                                                    width=image.shape[1],
                                                    count=1,
                                                    dtype=image.dtype,
                                                    crs='+proj=latlong') as dst: # non-georeferenced (just to be recognised from gis)
                            # Preprocessing before prediction
                            nan_mask = np.isnan(image)
                            image[nan_mask] = impute_nan[nan_mask]
                    
                            image = transform_test(image)
                            
                            image = standardization(image)
                            
                            image = image.unsqueeze(0)
                            
                            if options['test_time_augmentations']:
                                image = TTA(image) 
                            
                            # Image to Cuda if exist
                            image = image.to(device)
                    
                            all_predictions = []
                            for model in models_list:
                                # Predictions
                                logits = model(image)
                                logits = F.upsample(input=logits, size=(
                                    240, 240), mode='bilinear')
                        
                                predictions = torch.nn.functional.softmax(logits.detach(), dim=1)
                        
                                predictions = predictions.argmax(1)+1
                                
                                if options['test_time_augmentations']:
                                    predictions = TTA(predictions, reverse_aggregation = True)
                                all_predictions.append(predictions)
            
                            all_predictions = torch.cat(all_predictions)
                            all_predictions = torch.mode(all_predictions, dim=0, keepdim=True)[0]
                            
                            predictions = predictions.squeeze().cpu().numpy()
                            # Write the mask with georeference
                            dst.write_band(1, predictions.astype(dtype).copy()) # In order to be in the same dtype
                            dst.update_tags(**tags)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', help='Path of the images')
    parser.add_argument('--split', default = 'test', type = str, help='Which dataset split (test or val)')
    parser.add_argument('--test_time_augmentations', default= True, type=bool_flag, help='Generate maps and score based on multiple augmented testing samples? (Use batch = 1 !!!) ')

    parser.add_argument('--batch', default=1, type=int, help='Number of epochs to run')
	
    # Unet parameters
    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    parser.add_argument('--output_channels', default=15, type=int, help='Number of output classes')
    
    # Unet model path
    parser.add_argument('--model_path', default=os.path.join(up(os.path.abspath(__file__)), 'trained_models', '45', 'model_ema.pth'), help='Path to Unet pytorch model')
    parser.add_argument('--results_percentage', default= True, type=bool_flag, help='Generate confusion matrix results in percentage?')
    
    # Produce Predicted Masks
    parser.add_argument('--predict_masks', default= False, type=bool_flag, help='Generate test set prediction masks?')
    parser.add_argument('--gen_masks_path', default=os.path.join(root_path, 'data', 'predicted_marinext'), help='Path to where to produce store predictions')

    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict
    
    main(options)
