from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import RectifiedAdam


## my module
from tfa_AdaBelief import AdaBelief
# from tfa_RectifiedAdam import RectifiedAdam

class Optimizer:
    
    @staticmethod
    def decide_optimizer(optimizer="adam"):
        opt_func = None
        if (optimizer == "adam"):
            opt_func = Adam
        elif (optimizer == "adabelief"):
            opt_func = AdaBelief
        elif (optimizer in ["radam", "rectifiedadam"]):
            opt_func = RectifiedAdam
        
        return opt_func