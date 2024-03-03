import random
import cv2
import numpy as np

# my module
from colorPrint import Cprint as cp


## ================ config ================


WHITE_IMAGE = "/workspace/explain/onetime/white.png"
BLACK_IMAGE = "/workspace/explain/onetime/black.png"


## ========================================


class BCL:
    """
    @機能：
    @引数：
    @戻値：
    """
    
    def __init__():
        pass
    
    
    @staticmethod
    def data_division(resourcepath:str, random_seed:int=0, limit:int=None, is_include_fresh:bool=False, is_extend_luminance:bool=False):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        before_image_path, just_image_path = [], []
        before_mask_path, just_mask_path = [], []
        before_fresh_data, just_fresh_data = [], []
        image_path, mask_path, fresh_data, luminance_data = [], [], [], []
        labels = []
        
        with open(resourcepath) as f:
            readlines = f.readlines()
            random.seed(random_seed)
            random.shuffle(readlines)
            if limit: readlines = readlines[:limit]

        print()
        count = 0; length = len(readlines)
        
        for line in map(lambda x: x.rstrip("\n"), readlines):
            linebuffer = line.split(" ")
            image_path.append(linebuffer[0])
            mask_path.append(linebuffer[1])
            mask = np.array(cv2.imread(linebuffer[1]))
            mask = mask[:, :, 0]/255
            label = np.sum(mask) >= mask.shape[0]*mask.shape[1]/2
            labels.append(label)
            
            ## Just
            if label:
                just_image_path.append(linebuffer[0])
                just_mask_path.append(linebuffer[1])
                if is_include_fresh:
                    fresh_data.append(list(map(float, linebuffer[2:])))
                    just_fresh_data.append(list(map(float, linebuffer[2:])))
            ## Before
            else:
                before_image_path.append(linebuffer[0])
                before_mask_path.append(linebuffer[1])
                if is_include_fresh:
                    fresh_data.append(list(map(float, linebuffer[2:])))
                    before_fresh_data.append(list(map(float, linebuffer[2:])))
            if is_extend_luminance:
                luminance_data.append(float(linebuffer[-1]))
        
            cp.cprint(f"\033[1Aimage mixing {'.'*((count//50)%4)}{' '*(3-((count//50)%4))} | completed : {count+1} / {length} ( {round((count+1)/length*100, 2)} [%] ){' '*10}", "pink")
            count += 1
        
        mix_image_path = [None]*len(labels)
        
        for i, label in enumerate(labels):
            ## Just
            if label:
                path = random.choice(before_image_path)
            else:
                path = random.choice(just_image_path)
            mix_image_path[i] = path
        
        return image_path, mask_path, fresh_data, luminance_data, mix_image_path
