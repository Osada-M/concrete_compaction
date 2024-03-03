from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input


class HoG_Model:
    
    cell_size = 2
    block_size = 4
    dim = block_size**2
    
    
    @staticmethod
    def tuning(model, size=(576, 576)):
        
        hog_model = Sequential()
        hog_model.add(Input(shape=(*size, HoG_Model.dim)))
        
        for layer in model.layers[1:]:
            hog_model.add(layer)
        
        del model
        
        return hog_model
