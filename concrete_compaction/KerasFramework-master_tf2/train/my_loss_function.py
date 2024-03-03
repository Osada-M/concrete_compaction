from tensorflow.keras.losses import categorical_crossentropy, MeanSquaredError
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class MyLosses:
    
    @staticmethod
    def convert(ans, pred):
        """
        分散表現を局所表現に変換
        """
        ans = np.copy(ans.numpy()[0])
        pred = np.identity(4)[np.argmax(pred.numpy()[0], axis=2)]
        ans = np.uint8(ans)
        pred = np.uint8(pred)
        
        return ans, pred
    
    
    @staticmethod
    def iou(ans, pred):
        """
        IoUの算出
        """
        ans, pred = MyLosses.convert(ans, pred)
        
        if np.sum(pred | ans):
            iou_val = np.sum(pred & ans) / np.sum(pred | ans)
        else:
            return np.sum(pred & ans)
        
        return iou_val
    
    
    @staticmethod
    def iou_loss(ans, pred, is_priority:bool=True):
        """
        IoU Lossの算出
        """
        coefficient = 3. if is_priority else 2.
    
        return coefficient*(1 - MyLosses.iou(ans, pred))
    
    
    @staticmethod
    def ce_loss_debug(ans, pred):
        """
        Cross Entropyの算出(デバッグ用)
        """
        return categorical_crossentropy(ans, pred)

    
    @staticmethod
    def crossentropy_iou_loss(ans, pred):
        """
        Cross Entropy Loss + IoU Loss
        """
        ce_loss = categorical_crossentropy(ans, pred)
        iou_loss = MyLosses.iou_loss(ans, pred, False)
        
        return ce_loss + iou_loss
    
    
    @staticmethod
    def crossentropy_ssim_loss(ans, pred):
        """
        Cross Entropy Loss + SSIM Loss
        """
        ce_loss = categorical_crossentropy(ans, pred)
        ssim_loss = MyLosses.ssim_loss(ans, pred)
        
        return ce_loss + ssim_loss
    
    
    @staticmethod
    def rectified_mse_loss(ans, pred):
        """
        補正項付きMSE
        """
        
        mse = MeanSquaredError()
        mse_loss = mse(ans, pred)
        rec = 20.
        
        return rec * mse_loss
    
    
    @staticmethod
    def ssim_loss(ans, pred):
        ssim = 1-tf.reduce_mean(tf.image.ssim(ans, pred, max_val=1.0,filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03))
        rec = 1.
        
        return rec * ssim
    
    
    @staticmethod
    def ssim_mse_loss(ans, pred):
        ssim = 1-tf.reduce_mean(tf.image.ssim(ans, pred, max_val=1.0,filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03))
        mse = MeanSquaredError()
        mse_loss = mse(ans, pred)
        rec = [1., 10.]
        
        return rec[0]*ssim + rec[1]*mse_loss



class CrossEntropy_IoU(tf.keras.losses.Loss):
    """
    Cross Entropy Loss + IoU Loss
    コンストラクタ呼び出し用
    """
    def __init__(self, **kwargs):
        super(CrossEntropy_IoU, self).__init__(**kwargs)
        
    
    def get_config(self, **kwargs):
        return dict()
    
    
    def call(self, ans, pred):
        return MyLosses.crossentropy_iou_loss(ans, pred)
