import numpy as np
from PIL import Image


class AffineTransform:
    
    empty_label = 1 << 7
    r0 = None
    h, w = None, None
    first_point = None
    
    
    @staticmethod
    def rotate(image, radian, answer=False, fill=False, is_numpy=True):
        """
        Image rotate from affine transformation.
        """
        
        rotated = Image.fromarray(np.uint8(image))
        rotated = rotated.rotate(radian)
        if is_numpy:
            rotated = np.array(rotated)

        return rotated
    
    
    @staticmethod
    def point_rotate(a, b, radian):
        """
        Rotation for some coordinates that on 2-dimentional vector.
        """
        
        radian = np.radians(radian)
        matrix = np.array(
            [[np.cos(radian), -np.sin(radian)],
             [np.sin(radian), np.cos(radian)]]
        )
        a = matrix @ a
        b = matrix @ b
        
        return a, b
    
    
    @staticmethod
    def rotate_enlargement(image, radian, answer=False, fill=False):
        """
        Image rotate from affine transformation and adaptive enlargement.
        """
        
        if not radian%90:
            rotated = AffineTransform.rotate(image, radian, answer, fill, is_numpy=False)
            return np.array(rotated)
        
        rotated = AffineTransform.rotate(image, radian, answer, fill, is_numpy=False)
        
        ## Calculate each numbers from the original image.
        if AffineTransform.r0 is None:
            AffineTransform.h, AffineTransform.w = image.shape[:2]
            # if (AffineTransform.h == AffineTransform.w):
            #     AffineTransform.r0 = np.sqrt((AffineTransform.h/2)**2 + (AffineTransform.w/2)**2)
            # else:
            length = min([AffineTransform.h, AffineTransform.w])
            AffineTransform.r0 = length/2
            AffineTransform.first_point = [np.array([AffineTransform.w/2, AffineTransform.h/2], dtype=np.float32),
                                           np.array([-AffineTransform.w/2, AffineTransform.h/2], dtype=np.float32)]
            
        ## Calculate each coordinates top edge from the rotated image.
        a, b = AffineTransform.point_rotate(AffineTransform.first_point[0], AffineTransform.first_point[1], radian)
        a, b = AffineTransform.point_rotate(a, b, -(radian//90)*90)
        
        """note
        Define a and b are coordinates top edge from the rotated image. 
        
        (1) y = (dy/dx)x + I
        (2) g = -x
        dy/dx = (a.y - b.y) / (a.x - b.x)
        I = b.y + (dy/dx)|b.x|
        
        Solve x from the equation that f(x) = g(x).
        
        x = -1/(dy/dx + 1) * (b.y + (dydx)|b.x|)
        x = -((a.y - b.y)|b.x| + (a.x - b.x)b.y) / (a.x + a.y - b.x - b.y)
        
        The desired magnification (resctify_coef) is ||a|| / (x^2 + y^2)).
        From (2), Can transform the equation as ||a|| / (SQRT(2)*|x|).
        """
        
        x = abs( (((a[0]-b[0])*b[1]) - ((a[1]-b[1])*b[0])) / (a[0] + a[1] - b[0] - b[1]) )
        rectify_coef = AffineTransform.r0 / x
        
        # rotated = Image.fromarray(np.uint8(rotated))
        rotated = rotated.resize(size=(int(rectify_coef*AffineTransform.w), int(rectify_coef*AffineTransform.h)))
        rotated = rotated.resize(size=(int(AffineTransform.w), int(AffineTransform.h)),
                                 resample=Image.LANCZOS,
                                 box=(
                                     (rectify_coef-1)*AffineTransform.w//2,
                                     (rectify_coef-1)*AffineTransform.h//2,
                                     (rectify_coef-1)*AffineTransform.w//2 + AffineTransform.w,
                                     (rectify_coef-1)*AffineTransform.h//2 + AffineTransform.h,
                                     )
                                 )
        rotated = np.array(rotated)
        
        return rotated
    
    
    @staticmethod
    def enlargement(image, large):
        """
        Image enlargement.
        """
        
        h, w = image.shape[:2]
        larged = Image.fromarray(np.uint8(image))
        larged = larged.resize(size=(int(large*w), int(large*h)))
        larged = larged.resize(size=(int(w), int(h)),
                                 resample=Image.LANCZOS,
                                 box=(
                                     (large-1)*w//2,
                                     (large-1)*h//2,
                                     (large-1)*w//2 + w,
                                     (large-1)*h//2 + h,
                                     )
                                 )
        larged = np.array(larged)
        
        return larged


if(__name__ == "__main__"):
    
    import os
    from PIL import Image
    from tensorflow.keras.models import load_model

    ## my modules
    from MyUtils import ImageManager as im
    from MyUtils import Utils

    
    SAVE_ID = "fill_enlarge_learned"
    # SAVE_ID = "default"
    IMAGE_ID = "06_0150"
    IMAGE_ID = "06_1024"

    ORIGINAL_IMG = f"/workspace/Dataset/fullframe/image/190802/{IMAGE_ID}.png"
    ORIGINAL_MSK = f"/workspace/Dataset/fullframe/masked_4class/190802/masked_{IMAGE_ID}.png"
    SAVE_DIR = f"/workspace/visualization/affine_transform/{SAVE_ID}"
    
    # MODEL_ID = "e-unet_4classes_flip_ssim_10x-mse"
    MODEL_ID = "e-unet_4classes_flip_rotate_ssim_10x-mse"
    MODEL = f"/workspace/fullframe/result/robust_model/{MODEL_ID}_fold1/{MODEL_ID}_fold1.h5"
   
    COLORS = np.array([
        [255, 0, 0],
        [255, 64, 255],
        [64, 255, 255],
        [0, 255, 0],
    ], dtype=np.uint8)
    
    Utils.makedir(SAVE_DIR)
    Utils.makedir(f"{SAVE_DIR}/image")
    Utils.makedir(f"{SAVE_DIR}/mask")
    Utils.makedir(f"{SAVE_DIR}/pred")
    
    image = np.array(Image.open(ORIGINAL_IMG).resize((576, 576)))
    mask = np.array(Image.open(ORIGINAL_MSK).resize((576, 576)))
    model = load_model(MODEL)
    
      
    for radian in range(25):
        large = ((radian / 24) / 2) + 1
        radian *= 360 // 24
        
        # image_ = AffineTransform.rotate(image, radian, fill=True)
        image_ = AffineTransform.rotate_enlargement(image, radian)
        # image_ = AffineTransform.enlargement(image, large)
        input_image = np.reshape(image_ / 255., (1, 576, 576, 3))
        image_ = Image.fromarray(np.uint8(image_))
        image_.save(f"{SAVE_DIR}/image/{radian}.png")
        
        # mask_ = AffineTransform.rotate(mask, radian)
        mask_ = AffineTransform.rotate_enlargement(mask, radian)
        # mask_ = AffineTransform.enlargement(mask, large)
        mask_ = Image.fromarray(np.uint8(mask_))
        mask_.save(f"{SAVE_DIR}/mask/{radian}.png")
        
        pred ,= model.predict(input_image)
        pred = np.argmax(pred, axis=2)
        pred_image = np.zeros((576, 576, 3))
        for label in range(4):
            pred_image += COLORS[label] * np.repeat(np.reshape((pred==label), (*pred.shape, 1)), 3, axis=2)
        pred = Image.fromarray(np.uint8(pred_image))
        pred.save(f"{SAVE_DIR}/pred/{radian}.png")
        
        print(radian)
    
"""old
## dy/dx
# different = (a[1] - b[1]) / (a[0] - b[0])
## I
# intercept = b[1] + abs(b[0])*different        
# x = abs( intercept / ( different+1 ) )
"""