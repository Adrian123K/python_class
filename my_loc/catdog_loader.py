test_img = 'd:/catdog

def img_load(path):
    import os
    import re
    import cv2 
    import numpy as np 
    
    file_list = os.listdir(path)
    file_name = [int(re.sub('[^0-9]','',i)) for i in file_list]
    file_name.sort()
    f2 = [f'{path}/{i}.png' for i in file_name] 
    
    img = [cv2.imread(i) for i in f2] 
    
    return np.array(img) 
