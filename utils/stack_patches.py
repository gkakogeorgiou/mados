'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: stack_patches.py production of upsampling and stacking the patches.
'''

import os
import argparse
import shutil
import rasterio
from tqdm import tqdm
from glob import glob
from rasterio.enums import Resampling

def get_band(path, crop):
    return int(path.split('_'+crop)[0].split('_')[-1])

def main(options):
    
    main_output_folder = options['path'] + '_' + options['resampling']
    all_tiles = glob(os.path.join(options['path'],'Scene_*'))
    if options['resampling'] == 'nearest':
        resampling_method = Resampling.nearest
    elif options['resampling'] == 'bilinear':
        resampling_method = Resampling.bilinear
    else:
        raise

    # Copy split folder
    split_output_folder = os.path.join(main_output_folder, 'splits')
    os.makedirs(split_output_folder, exist_ok=True)
    
    split_files = glob(os.path.join(options['path'],'splits','*'))
    for f in split_files:
        new_split_file = os.path.join(split_output_folder, os.path.basename(f))
        shutil.copy(f, new_split_file)

    for tile in tqdm(all_tiles):
        
        # Create the output folder
        current_output_folder = os.path.join(main_output_folder, os.path.basename(tile))
        os.makedirs(current_output_folder, exist_ok=True)
        
        # Copy gt files
        gt_files = glob(os.path.join(tile, '10', '*_cl_*')) + glob(os.path.join(tile, '10', '*_conf_*')) + glob(os.path.join(tile, '10', '*_rep_*'))
        for f in gt_files:
            new_gt_file = os.path.join(current_output_folder, os.path.basename(f))
            shutil.copy(f, new_gt_file)
        
        # Get the number of different crops for the specific tile
        splits = [f.split('_cl_')[-1] for f in glob(os.path.join(tile, '10', '*_cl_*'))]
        
        for crop in splits:
            
            # Get the bands for the specific crop 
            all_bands = glob(os.path.join(tile, '*', '*L2R_rhorc*_'+crop))
            all_bands = sorted(all_bands, key=lambda patch: get_band(patch, crop))
            
            ################################
            # Stack and Upsample the bands #
            ################################
            
            # Get metadata from the second 10m band
            with rasterio.open(all_bands[1], mode ='r') as src:
                tags = src.tags().copy()
                meta = src.meta
                image = src.read(1)
                shape = image.shape
                dtype = image.dtype
    
            # Update meta to reflect the number of layers
            meta.update(count = len(all_bands))
    
            # Construct the filename
            output_file = os.path.basename(all_bands[1]).replace(str(get_band(all_bands[1], crop))+'_', '')
            output_file = os.path.join(current_output_folder, output_file)
    
            # Write it to stack
            with rasterio.open(output_file, 'w', driver='GTiff',
                                                height=shape[-2],
                                                width=shape[-1],
                                                count=len(all_bands),
                                                dtype=dtype,
                                                crs='+proj=latlong') as dst: # non-georeferenced (just to be recognised from gis)
                
                for c, band in enumerate(all_bands, 1):
                    upscale_factor = int(os.path.basename(os.path.dirname(band)))//10
    
                    with rasterio.open(band, mode ='r') as src:
                        dst.write_band(c, src.read(1,
                                                        out_shape=(
                                                            int(src.height * upscale_factor),
                                                            int(src.width * upscale_factor)
                                                        ),
                                                        resampling=resampling_method
                                                      ).astype(dtype).copy()
                                      )
                dst.update_tags(**tags)    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--path', help='Path to dataset')
    parser.add_argument('--resampling', default='nearest', type=str, help='Type of resampling before stacking (nearest or bilinear)')
    
    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict
    
    main(options)
