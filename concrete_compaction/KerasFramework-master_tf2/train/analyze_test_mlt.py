# -*- coding: utf-8 -*-
# import library
import csv
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from sklearn.metrics import classification_report


import re
import pandas as pd
import numpy as np
import time
import yaml
import sys
import os
import h5py
import matplotlib.pyplot as plt
import cv2
import argparse
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from PIL import Image 

from datasetgenerator.DatasetGenerator import DatasetGenerator
from datasetgenerator.ConcLocalProcesser import ConcLocalProcesser
from mymodel.CreateModel import CreateModel


def cl_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("conf", help="config yaml file")
    args = parser.parse_args()
    print("config file: ", args.conf)
    return args.conf


# make function
def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def print_and_file_write(file, string):
    file.write(string + "\n")
    print(string)

''' OpenCV => PIL '''
def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image




# open congig yaml file.
print("open config file...")

arg = cl_arg_parser()

with open(arg) as file:
    print("complete!")
    yml = yaml.safe_load(file)


# aveimg_path = '/workspace/concrete_compaction/aveimg.jpg'
# aveimg_path = '/media/nagalab/Volume01/kojima_ws/concrete_compaction/average_image_0516.png'
aveimg_path = '/workspace/kojima_ws/average_image_0516.png'
#ave_image = cv2.imread(aveimg_path)
#ave_image = cv2.cvtColor(cv2.imread(aveimg_path), cv2.COLOR_BGR2RGB)/255.0
ave_image= img_to_array(Image.open(aveimg_path))/255.0


# Dataset Generator loadding.

#------------------------------------------------------------

dataset = DatasetGenerator()

print("test dataset loading...")
print("--------- dataset ---------")
if yml["testresourcedata"]["readdata"] == "text" or yml["testresourcedata"]["readdata"] == "TEXT":
    if yml["SemanticSegmentaiton"]["available"]:
        testdata = dataset.text_segmentation_dataset(yml["SemanticSegmentation"]["resourcedata"]["test"])
    else:
        testdata = dataset.text_dataset(yml["testresourcedata"]["resourcepath"])
elif yml["testresourcedata"]["readdata"] == "onefolder" or yml["testresourcedata"]["readdata"] == "Onefolder":
    testdata = dataset.onefolder_dataet(yml["testresourcedata"]["resourcepath"])
elif yml["testresourcedata"]["readdata"] == "folder" or yml["testresourcedata"]["readdata"] == "Folder":
    testdata = dataset.folder_dataset(yml["testresourcedata"]["resourcepath"])
else:
    print("It appears that you have selected a data loader that is not specified. Stops the program.")

#---------------------------------
conc_gen = ConcLocalProcesser()
images, labels = [], []
for d in testdata:
    images.append(d[0])
    labels.append(d[1])
datacount = len(testdata)
freshes = conc_gen.extract_freshdata(testdata)
print("test data : ", datacount)
print("---------------------------")
print("---------- class ----------")
classes = yml["Resourcedata"]["classes"]
for c in classes:
    print(c)
print("---------------------------")



"""
ar_freshes = np.asarray(freshes, dtype=np.float32)
ar_freshes[0].shape
"""



# plot dataset
xdata, ydata = [], []

# load dataset... and read PIL image.
for x in range(25):
    rand_num=np.random.randint(0,datacount)
    origin_image = img_to_array(load_img(images[rand_num], color_mode="rgb", target_size=(yml["Resourcedata"]["img_row"], yml["Resourcedata"]["img_col"])))/255.0
    subed_image = origin_image - ave_image
    xdata.append(subed_image)
    #xdata.append(img_to_array(load_img(images[rand_num], color_mode="rgb", target_size=(yml["Resourcedata"]["img_row"], yml["Resourcedata"]["img_col"])))/255)
    ydata.append(labels[rand_num])

# image plot
plt.figure(figsize=(25,25), facecolor='white')
for i in range(25):
    cifar_img=plt.subplot(5,5,i+1)
    plt.imshow(abs(xdata[i]))
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.title(classes[int(ydata[i])])


# ---
# 
# 
# 



print("---------  model  ---------")
createModel = CreateModel(yml)
# open congig yaml file.
# model = load_model(yml["TESTModel"]["path"] + yml["TESTModel"]["model_path"])
model = createModel.create_model('resnet', num_feature=2)
#model = model_from_json(open(yml["TESTModel"]["path"] + yml["TESTModel"]["model_path"]).read())

print("Load model weight...")
model.load_weights(yml["TESTModel"]["path"] + yml["TESTModel"]["weight_path"])

"(6/18) ArcFace, CosFace, SphereFaceの時の処理. TODO: outputsをlayer[-4]で実行"
# TODO:  batch_normalization層って何してるん?
"layer[-3]: BatchNormalize... ?(7/16: -3がdense説ワンちゃんあるよな."
# model = Model(inputs=model.input[:2], outputs=model.layers[-3].output)
"layer[-4]: Dense"
# model = Model(inputs=model.input[:2], outputs=model.layers[-4].output)
# model = Model(inputs=model.input[:2], outputs=model.get_layer('classifer').output)

model.summary()

print("---------------------------")

"""
hoge = np.asarray(freshes, dtype=np.float32)
print(freshes[0])
print(hoge[0])
ar_freshes[0].shape
"""



#ar_freshes = np.asarray(freshes, dtype=np.float32)
#print(ar_freshes[0].shape)

predict_list, ans_list, predict_rawdata = [], [], []


for count, (x, ans) in enumerate(zip(tqdm(images), labels)):
    # image file open.
    try:
        # I'll load the image, and if it doesn't work, I'll terminate the program.
        #image = img_to_array(load_img(x, color_mode="rgb", target_size=(yml["Resourcedata"]["img_row"], yml["Resourcedata"]["img_col"])))
        image = img_to_array(load_img(x, color_mode="rgb", target_size=(yml["Resourcedata"]["img_row"], yml["Resourcedata"]["img_col"])))
        # Normalize image.
        image /= 255.0
        image = image - ave_image
    except Exception as e:
        print("Failed to load data.")
        print("ERROR : ", e)

    # model predict => pred_label
    image = np.asarray([image], np.float32)
    #fresh = np.asarray([fresh], dtype=np.float32)
    fresh = np.asarray([freshes[count]], np.float32)
    #print(type(ar_freshes[count]))
    #print(ar_freshes[count].shape)
    #print(image.shape)
    predict = model.predict([image, fresh], batch_size=1)
    predict /= np.linalg.norm(
        predict,
        axis=1,
        keepdims=True
    )
    
    predict_rawdata.append(predict)
    
    predict_list.append(np.argmax(predict))
    # ans => answer list
    ans_list.append(int(ans))

# Raw data of predicted values
#print(predict)




# calassification report. and write csv

result = classification_report(ans_list, predict_list, target_names = classes, output_dict=True)
print("--------------< result >--------------")
print(classification_report(ans_list, predict_list, target_names = yml["Resourcedata"]["classes"]))
print("--------------------------------------")
class_pd = pd.DataFrame(result)
class_pd.to_csv(yml["TESTModel"]["path"] + "/classification_report.csv")


# 



# correct list and uncorrect list creat.
print("correct and fail predict list create...")

correct_list, uncorrect_list ,combined_list= [], [],[]
for (image_path, prd, ans, raw_prd) in zip(images, predict_list, ans_list, predict_rawdata):
    if ans == prd:
        correct_list.append([image_path, ans, prd, str(raw_prd.tolist())])
    else:
        uncorrect_list.append([image_path, ans, prd, str(raw_prd.tolist())])
    combined_list.append([image_path, ans, prd, str(raw_prd.tolist())])
correct_df = pd.DataFrame(correct_list, columns=['inputdata', 'answer', 'predict', 'predict_rawdata'],index=range(len(correct_list)))
uncorrect_df = pd.DataFrame(uncorrect_list, columns=['inputdata', 'answer', 'predict', 'predict_rawdata'],index=range(len(uncorrect_list)))
combined_df = pd.DataFrame(combined_list, columns=['inputdata', 'answer', 'predict', 'predict_rawdata'],index=range(len(combined_list)))

#print(correct_df)
#print(uncorrect_df)

print("result write csv...")
correct_df.to_csv(yml["TESTModel"]["path"] + "/predict_correct_list.csv")
uncorrect_df.to_csv(yml["TESTModel"]["path"] + "/predict_uncorrect_list.csv")
combined_df.to_csv(yml["TESTModel"]["path"] + "/predict_combined_list.csv")
print("complate")



def create_graph(reader):
    fig, ax = plt.subplots()
    correct_label = [[] for i in range(24)]
    pred_result = [[] for i in range(24)]  #正解ラベルと実験結果が一致しているかしていないか
    mesh_index_list = []

    for row in reader:
        image_name = row[1].split('/')[-1]
        image_name = image_name.split('.')[0]
        video_num = image_name.split('_')[-1]

        mesh_num = row[1].split('/')[-1]
        mesh_num = mesh_num.split('.')[0]
        mesh_index = mesh_num.split('_')[-2]
        mesh_index_list.append(mesh_index)
        if video_num == '01':
            len_index_list = int(len(mesh_index_list))

        sample_day = image_name.split('_')[0]
        sample_num = image_name.split('_')[1]

        mesh_index = int(video_num)-1  

        #for i in row:
        if row[2] == row[3]:
            pred_result[mesh_index].append('true')
        else:
            pred_result[mesh_index].append('false')

        if row[2] == '0':
            correct_label[mesh_index].append('before')
            
            #print(counter)
        else:
            correct_label[mesh_index].append('just')
        #print(correct_label)
        #print(np.array(correct_label).shape)

    for i in range(24):
        index_before = [i for i, x in enumerate(correct_label[i]) if x == 'before']
        index_before = list(zip(index_before,np.ones(len(index_before))))
        #print(index_before)
        index_just = [i for i, x in enumerate(correct_label[i]) if x == 'just']
        index_just = list(zip(index_just,np.ones(len(index_just))))
        #print(index_just)
        index_true = [i for i, x in enumerate(pred_result[i]) if x == 'true']
        index_true = list(zip(index_true,np.ones(len(index_true))))
        #print(index_true)
        index_false = [i for i, x in enumerate(pred_result[i]) if x == 'false']
        index_false = list(zip(index_false,np.ones(len(index_false))))
        #print(index_false)
        #index_mesh = [i for i, x in enumerate(mesh_index[i]) if x == 'empty']
        #index_mesh = list(zip(index_mesh,np.ones(len(index_mesh))))

        yrange1 = [i+1, 0.5]
        yrange2 = [i+0.5, 0.5] 

        ax.broken_barh(xranges=index_false, yrange=yrange2, facecolor='black')
        ax.broken_barh(xranges=index_true,yrange=yrange2, facecolor='white')
        ax.broken_barh(xranges=index_before, yrange=yrange1, facecolor='red')
        ax.broken_barh(xranges=index_just,yrange=yrange1, facecolor='green')
        #ax.broken_barh(xranges=index_just,yrange=yrange1, facecolor='purple')
        ax.set_yticks(list(range(1, 25)))
        ax.set_xlim([0, len_index_list])

    plt.savefig(yml["TESTModel"]["path"] + sample_day +"_"+ sample_num+ "result.png")
    plt.title(sample_day + "_" + sample_num)

def extract_video_num(row):
    image_name = row[1].split('/')[-1]
    image_name = image_name.split('.')[0]
    video_num = image_name.split('_')[-3]
    video_num = video_num[:-2]      #名前を分割後、abをなくす
    return video_num

def day_num_video_list(row):
    image_name = row[1].split('/')[-1]
    image_name = image_name.split('.')[0]
    video_num = image_name.split('_')[-3]
    video_num = video_num[:-2]      #名前を分割後、abをなくす
    day_num = image_name.split('_')[0]
    day_num_video = day_num + '_' + video_num     #動画の日付と番号をまとめたもの
    return day_num_video




mesh_list = []
local_index = 0
i = 0
csv_path = yml["TESTModel"]["path"] + "/predict_combined_list.csv"
with open(csv_path) as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in tqdm(reader):
        video_num = day_num_video_list(row)
        #print(video_num)
        mesh_list.append(video_num)   #リストにする
        mesh_set = set(mesh_list)  #集合にしている
        mesh_set_list = list(mesh_set)  #集合をリストにする
        sort_mesh_set_list = sorted(mesh_set_list)               #リストを昇順にしている
    #print(mesh_set_list)
    #画像データごとにcsvファイルを分割
    f.seek(0)
    reader = csv.reader(f)
    header = next(reader)
    split_result = []   #split_resultをmesh_set_resultの長さだけのリストにする 例）長さが２の場合[0,1]
    split_result = [[] for _ in range(len(sort_mesh_set_list))]
    for row in tqdm(reader):
        video_num2 = extract_video_num(row)     #行を読み込んだ時のその行の動画番号を抽出
        if video_num2 != local_index:
            i += 1
            local_index = video_num2
        split_result[i-1].append(row)
    for i in range(len(sort_mesh_set_list)):
        create_graph(split_result[i])




# show detail result. 
# Indicate which items are being answered and by how much.
category_result = [0] * len(classes)
class_count = 0

# for result write file.
with open(yml["TESTModel"]["path"] + "/predict_result.txt", mode='w') as f:
    print_and_file_write(f, "data count : {}".format(datacount))

    for category in range(len(classes)):
        print_and_file_write(f, "============{}================".format(classes[category]))
        # Check your predict and answers.
        for (prd, ans) in zip(predict_list, ans_list):
            # If the answer and the category (rotating in a for loop) match, add the number of categories inferred
            if category == ans:
                category_result[prd] += 1
                class_count += 1
        # Show the results of the inference.
        for c, prd_count in enumerate(category_result):
            print_and_file_write(f ,"answer = {}, predict {} => {}".format(classes[category], classes[c], prd_count))
        # Show the number of each category
        print_and_file_write(f, "class data count : {}".format(class_count))
        # reset result list. and reset class count
        category_result = [0] * len(classes)
        class_count = 0 
    print_and_file_write(f, "=====================================")
