# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:05:17 2020

@author: 11095
"""

import numpy as np
from PIL import Image

def create_cityscapes_label_colormap():
      """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    
      Returns:
        A colormap for visualizing segmentation results.
      """
      colormap = np.zeros((256, 3), dtype=np.uint8)
      colormap[0] = [0, 0, 0]#粉
      colormap[1] = [0, 0, 255]#蓝
      colormap[2] = [0, 255, 0]#绿
      colormap[3] = [255, 255, 0]#黄
      colormap[4] = [255, 0, 0]    #红  
      colormap[5] = [0, 0, 0] 
      
#      colormap[6] = [250, 170, 30]
#      colormap[7] = [220, 220, 0]
#      colormap[8] = [107, 142, 35]
#      colormap[9] = [152, 251, 152]
#      colormap[10] = [70, 130, 180]
#      colormap[11] = [220, 20, 60]
#      colormap[12] = [255, 0, 0]
#      colormap[13] = [0, 0, 142]
#      colormap[14] = [0, 0, 70]

      return colormap
    
    
def label_to_color_image(label):
      """Adds color defined by the dataset colormap to the label.
      """
      if label.ndim != 2:
        raise ValueError('Expect 2-D input label. Got {}'.format(label.shape))
    
      colormap = create_cityscapes_label_colormap()
      return colormap[label]
  
    
def blend_img_colorlabel(img,label):  
    color_label=label_to_color_image(label)
    lab_image = Image.fromarray(color_label)#训练
    image = Image.fromarray(img)#训练
    _im=image.convert('RGBA')#原始image 4通道
    p_im=lab_image.convert('RGBA') #训练标签 4通道

    imgs = Image.blend(_im,p_im,0.6)

    return imgs