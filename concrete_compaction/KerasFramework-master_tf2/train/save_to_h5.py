import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
import umap
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from functools import lru_cache, partial

## my module
from mymodel.SemanticSegmentation import SemanticSegmentation
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
import MetricLearning_for_semseg as metric_semseg
from line_sender import send_master
from mymodel.SemSegLight import SemSegLight


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.get_logger().setLevel("ERROR")


MODEL = "/workspace/fullframe/result/540x540/e-unet_20220629_AutoLearning_fold5_576x576"

model = load_model(MODEL)
model.save(f"{MODEL}/e-unet_softmax_f5.h5")