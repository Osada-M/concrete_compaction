import numpy as np
import math


class LuminanceExtender:
    
    def __init__(self, shape:list):
        self.shape = shape
        self.row, self.col = shape
        
        ## 重ねる円
        origin_x, origin_y = self.col/2, self.row/2
        radius_pow = self.col*self.row/(2*math.pi)
        self.circle_img = np.zeros((self.row, self.col, 3))
        for y in range(self.row):
            for x in range(self.col):
                if ((x-origin_x)**2 + (y-origin_y)**2 <= radius_pow):
                    self.circle_img[y, x] = 1

        ## 重ねる斜め領域
        self.slant_img = np.zeros((self.row, self.col, 3))
        dydx = self.row / self.col
        for y in range(self.row):
            for x in range(self.col):
                if (y <= dydx * x):
                    self.slant_img[y, x] = 1

        ## 重ねる半分割領域
        self.half_img = np.zeros((self.row, self.col, 3))
        self.half_img[:, self.col//2:] += 1
    
    
    def extend(self, img, mode:str="none", extend_const:int=50, noise_type:str="linear"):
        """
        輝度の拡張(利用時はこのメソッドを呼び出す)
        """
        if (mode != "none"):
            # buf = np.array(img, dtype=np.int32)
        
            # funcs = {"circle" : self.circle,
            #         "half" : self.half_right,
            #         "all" : self.all_pixel,
            #         "slant" : self.half_slant}
            
            # result = funcs[mode](buf, extend_const)

            
            if (noise_type == "tanh"):
                buf = self.tanh(img, mode, extend_const)
            
            else:
                buf = self.linear(img, mode, extend_const)
            
            return np.uint8(np.clip(buf, 0, 255))
        
        else:
            return img
    
    
    def linear(self, img, mode, extend_const):
        """
        線形
        """
        img = np.float32(img)
        
        if (mode == "circle"):
            img += self.circle_img*extend_const
        
        elif (mode == "slant"):
            img += self.slant_img*extend_const
        
        elif (mode == "half"):
            img[:, self.col//2:] += extend_const
            
        elif (mode == "all"):
            img += extend_const
            
        return img
    
    
    def tanh(self, img, mode, extend_const):
        """
        非線形(tanh)
        """
        extend_const += 100
        extend_const /= 200
        extend_const *= 1.5
        
        if (mode == "circle"):
            change_area = np.copy(img * self.circle_img)
        
        elif (mode == "slant"):
            change_area = np.copy(img * self.slant_img)
        
        elif (mode == "half"):
            change_area = np.copy(img * self.half_img)
            
        elif (mode == "all"):
            change_area = np.copy(img)
        
        img = np.float32(img)
        change_area = np.float32(change_area)
        img *= change_area == 0.
        
        img += 255. * (np.tanh(5 * (change_area/255. - 0.5) + extend_const) + 1.) / 2.
        
        return img
    
    # def circle(self, img, extend_const:int=50):
    #     """
    #     円
    #     """
    #     img = np.uint8(np.clip(img + (self.circle_img*extend_const), 0, 255))
        
    #     return img


    # def half_slant(self, img, extend_const:int=50):
    #     """
    #     斜め
    #     """
    #     img = np.uint8(np.clip(img + (self.slant_img*extend_const), 0, 255))
        
    #     return img


    # def all_pixel(self, img, extend_const:int=50):
    #     """
    #     全ピクセル
    #     """
    #     img += extend_const
    #     img = np.uint8(np.clip(img, 0, 255))
        
    #     return img


    # def half_right(self, img, extend_const:int=50):
    #     """
    #     半分
    #     """
    #     img[:, self.col//2:] += extend_const
    #     img = np.uint8(np.clip(img, 0, 255))
        
    #     return img

