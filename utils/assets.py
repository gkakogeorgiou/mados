# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: assets.py includes the appropriate mappings.
'''
import numpy as np

mados_cat_mapping =  {'Marine Debris': 1,
					  'Dense Sargassum': 2,
					  'Sparse Floating Algae': 3,
					  'Natural Organic Material': 4,
					  'Ship': 5,
					  'Oil Spill': 6,
					  'Marine Water': 7,
					  'Sediment-Laden Water': 8,
					  'Foam': 9,
					  'Turbid Water': 10,
					  'Shallow Water': 11,
					  'Waves & Wakes': 12,
					  'Oil Platform': 13,
					  'Jellyfish': 14,
					  'Sea snot': 15}
				   
mados_color_mapping =  { 'Marine Debris': 'red',
						 'Dense Sargassum': 'green',
						 'Sparse Floating Algae': 'limegreen',
						 'Natural Organic Material': 'brown',
						 'Ship': 'orange',
						 'Oil Spill': 'thistle',
						 'Marine Water': 'navy',
						 'Sediment-Laden Water': 'gold',
						 'Foam': 'purple',
						 'Turbid Water': 'darkkhaki',
						 'Shallow Water': 'darkturquoise',
						 'Waves & Wakes': 'bisque',
						 'Oil Platform': 'dimgrey',
						 'Jellyfish': 'hotpink',
						 'Sea snot': 'yellow'}

labels = ['Marine Debris', 'Dense Sargassum', 'Sparse Floating Algae', 'Natural Organic Material', 
'Ship', 'Oil Spill', 'Marine Water', 'Sediment-Laden Water', 'Foam', 
'Turbid Water', 'Shallow Water', 'Waves & Wakes', 'Oil Platform', 'Jellyfish', 'Sea snot']

s2_mapping = {'nm440': 0,
              'nm490': 1,
              'nm560': 2,
              'nm665': 3,
              'nm705': 4,
              'nm740': 5,
              'nm783': 6,
              'nm842': 7,
              'nm865': 8,
              'nm1600': 9,
              'nm2200': 10,
              'Class': 11,
              'Confidence': 12,
              'Report': 13}

conf_mapping = {'High': 1,
                'Moderate': 2,
                'Low': 3}

report_mapping = {'Very close': 1,
                  'Away': 2,
                  'No': 3}

def cat_map(x):
    return mados_cat_mapping[x]

cat_mapping_vec = np.vectorize(cat_map)

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
		
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule