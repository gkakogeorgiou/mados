# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: test_time_aug.py includes the test-time eight augmentations
                              with random rotations and horizontal flips.
'''


import torch
from torchvision.transforms.functional import hflip

def TTA(img, reverse_aggregation = False):
    im_list = []
    
    if not reverse_aggregation:
        for k in [0,1,2,3]:
            im = torch.rot90(img, k=k, dims=[-2, -1])
            im_list.append(im)
            
            im = hflip(im)
            im_list.append(im)
            
        img = torch.cat(im_list)
        
    else:

        for k in [3,2,1,0]:
            im = hflip(img[k*2 + 1,:,:])
            im = torch.rot90(im, k=-k, dims=[-2, -1])
            im_list.append(im)
            
            im = torch.rot90(img[k*2,:,:], k=-k, dims=[-2, -1])
            im_list.append(im)

        img = torch.stack(im_list)
        
        img = torch.mode(img, dim=0, keepdim=True)[0]
    
    return img