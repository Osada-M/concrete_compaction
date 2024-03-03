from tensorflow.keras.layers import PReLU

# from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.sparsity.keras import PrunableLayer


# class PL_PReLU(PReLU, prunable_layer.PrunableLayer):
class PL_PReLU(PReLU, PrunableLayer):
    def get_prunable_weights(self):
        return self.forward_layer._trainable_weights + self.backward_layer._trainable_weights


def prune_rectify():
    
    prune_registry = "/usr/local/lib/python3.8/dist-packages/tensorflow_model_optimization/python/core/sparsity/keras/prune_registry.py"
    with open(prune_registry) as f:
        lines = f.read()

    if ("layers.PReLU" in lines): return False
    
    rectified = lines.replace("""layers_compat_v1.BatchNormalization: [],""",
    """layers_compat_v1.BatchNormalization: [],
      ## Modified by Masashi OSADA (NAGATA-Lab, 2022/12/01) 
      layers.PReLU: [],""")

    with open(prune_registry, mode="w") as f:
        f.write(rectified)
    
    return True


# if(__name__ == "__main__"):
#     import tensorflow_model_optimization as tfmot
#     print(tfmot)


"""
from tensorflow.python.keras.engine.functional import Functional
"""