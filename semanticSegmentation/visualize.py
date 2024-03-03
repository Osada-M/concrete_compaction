# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import pandas as pd
# import random
# from tqdm import tqdm
# from sklearn.metrics import classification_report
# import time
# import sys

# # conc modules
# from mymodel.semanticSegmentation import semanticSegmentation
# # my module
# from colorPrint import Cprint as cp


# ## ================ config ===================


# WORKSPACE_DIR = "/workspace/semanticSegmentation"
# DATASET_DIR = "/workspace/Dataset/semanticSegmentation"

# # BATCH_SIZE : 1 only
# BATCH_SIZE = 1
# # EPOCHS = 10

# FOLD = 1
# MODEL_NAME = "unet_1"
# SIZE = 270
# NUM_CLASSES = 2
# CLASSES = ["before", "just"]
# IS_MLT = False

# RANDOM_SEED = 1
# LIMIT = None
# USE_MULTI_GPU = False
# IS_RESIZE = False
# BUF = ""

# IS_OUTPUT_RESULT = True

# LOAD_PATH = "/workspace/semanticSegmentation/result"
# LOAD_ID = "unet_20220205"
# RESULT_ID = f"{LOAD_ID}_testResult"

# IS_VISUALIZE = False

# ## ===========================================



# text = lambda id : f"/workspace/Dataset/semanticSegmentation/text_dataset/visualize/take_image{id}.txt"

# image, mask = [], []
# with open(text(""), mode="r") as f:
#     lines = f.readline()
# for line in lines:
#     buffer = line.split(" ")
#     image.append(buffer[0])
#     mask.append(buffer[1].rstrip("\n"))


# def get_palette():
#     palette = [[0, 0, 0],
#                [255, 255, 255]]
#     return np.asarray(palette)


# def adjustData(img, mask):
#     if(np.max(img) > 1):
#         img = img / 255.

#     palette = get_palette()

#     onehot = np.zeros((mask.shape[0], SIZE, SIZE, NUM_CLASSES), dtype=np.uint8)
#     for i in range(2):
#         cat_color = palette[i]
#         temp = np.where((mask[:, :, :, 0] == cat_color[0]) &
#                         (mask[:, :, :, 1] == cat_color[1]) &
#                         (mask[:, :, :, 2] == cat_color[2]), 1, 0)
#         onehot[:, :, :, i] = temp

#     return img, onehot


# def dataGenerator(resourcepath):
    
#     image_path, mask_path = [], []
#     with open(resourcepath) as f:
#         readlines = f.readlines()
#         random.seed(RANDOM_SEED)
#         random.shuffle(readlines)
#         if LIMIT: readlines = readlines[:LIMIT]
#     for line in readlines:
#         linebuffer = line.split(" ")
#         image_path.append(linebuffer[0])
#         mask_path.append(linebuffer[1].rstrip("\n"))
            
#     data_gen_args = dict(
#         rescale=None
#     )

#     image_datagen = ImageDataGenerator(**data_gen_args)
#     mask_datagen = ImageDataGenerator(**data_gen_args)

#     image_dataframe = pd.DataFrame(image_path, index=None, columns=["image"])
#     mask_dataframe = pd.DataFrame(mask_path, index=None, columns=["mask"])

#     image_generator = image_datagen.flow_from_dataframe(image_dataframe,
#                                                         x_col="image",
#                                                         target_size=[SIZE, SIZE],
#                                                         color_mode="rgb",
#                                                         classes=["before", "just"],
#                                                         class_mode=None,
#                                                         batch_size=BATCH_SIZE,
#                                                         shuffle=False)
#     mask_generator = mask_datagen.flow_from_dataframe(mask_dataframe,
#                                                       x_col="mask",
#                                                       target_size=[SIZE, SIZE],
#                                                       color_mode="rgb",
#                                                       classes=["before", "just"],
#                                                       class_mode=None,
#                                                       batch_size=BATCH_SIZE,
#                                                       shuffle=False)
    
#     for image, mask in zip(image_generator, mask_generator):
#         image, mask = adjustData(image, mask)
#         yield image, mask    


# def datacounter(datapath):
#     with open(datapath) as f:
#         readlines = f.readlines()
#         if LIMIT : readlines = readlines[:LIMIT]
#     return len(readlines)


# def createModel(model_name:str=None):
    
#     model = None
    
#     if model_name is None:
#         blank("model_name")
            
#     else:
#         cp.cprint(f"- model : {model_name} -", "cyan")        
#         if (model_name == "unet"):
#             model = semanticSegmentation.unet([SIZE, SIZE, 3], NUM_CLASSES, IS_MLT)        
#         elif (model_name == "unet_1"):
#             model = semanticSegmentation.unet_1([SIZE, SIZE, 3])
#         else:
#             cp.cprint(f"[!] {model_name} is not defined.", "red")
        
#     return model    