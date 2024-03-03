from PIL import Image
import csv
import openpyxl
import numpy as np
import time

from colorPrint import Cprint as cp
from MyUtils import Utils


## ================ config ================


MESH = (4, 6)
# SIZE = (576, 576)
SIZE = (270*4, 270*6)
COLORS = [0, 128, 255]

SAVE_DIR = "/workspace/Dataset/fullframe/masked_after"
XLSX = "/workspace/osada_ws/220301正解ラベル_全データ.xlsx"
# DATE = ["190731", "190802", "190807", "190809"]
DATE = ["190807"]
IPT_CSV = "/workspace/mesh_dataset/ImagePathTime.csv"
HEADER_ROW = 4


## ========================================


xlsx_column_buf = lambda col: [0]*col


def str_to_time(string):
    
    if string is None:
        return 1e10
        
    time = str(string).split(":")
    result = 0
    for i, t in enumerate(time[::-1]):
        result += int(float(t)) * (60**i) * 30
        if i: break
    
    return result


def make_after_labeled_image():
    """
    Extract labels from excel.
    and
    Make label images.
    """
    
    xlsx = openpyxl.load_workbook(XLSX, data_only=True)
    dates = xlsx.sheetnames
    cp.cprint(f"\n{dates}\n\n↓↓↓\n", "cyan")
    dates = [s for s in dates if s in DATE]
    cp.cprint(f"{dates}\n", "cyan")
    
    mins, maxs = dict(), dict()
    min_buf = [None]*20
    max_buf = [None]*20
    old_key = ""
    
    ## Extract labels from excel.
    with open(IPT_CSV) as f:
        lines = f.readlines()[1:]
        for line in lines:
            key, b_min, _, _, j_max = (line.replace("\n", "")).split(",")
            key, video_id = key.split("_")[:-1]
            
            if not key in mins:
                mins[key], maxs[key] = [], []
            if (old_key != key):
                if min_buf[0] is not None:
                    mins[old_key] = [-1 if v is None else int(v) for v in min_buf]
                    maxs[old_key] = [-1 if v is None else int(v) for v in max_buf]
                min_buf = [None]*20
                max_buf = [None]*20
            
            old_key = key
            min_buf[int(video_id)-1] = b_min
            max_buf[int(video_id)-1] = j_max
            
        mins[key] = [-1 if v is None else int(v) for v in min_buf]
        maxs[key] = [-1 if v is None else int(v) for v in min_buf]
        
    masks = dict()
    
    ## Convert time to label array
    for date in dates:
        sheet = xlsx[date].values
        
        masks[date] = []
        mask = None
        for row, val in enumerate(sheet):
            if (row <= HEADER_ROW):
                continue
            
            video_id = val[0]
            mesh_id = val[2]
            start = val[3]
            end = val[4]
            just = val[18]
            after = val[34]
            
            start, end, just, after = map(str_to_time, [start, end, just, after])
            mesh_id -= 1
            
            if video_id is not None:
                if mask is not None:
                    masks[date].append(mask)
                mask = np.zeros((*MESH, 2))
            
            move = (end - start) / 30
            
            if (just >= end): just -= move
            if (after >= end): after -= move
            
            img_row = mesh_id // MESH[1]
            img_col = mesh_id % MESH[1]
            
            mask[img_row, img_col, 0] = just
            mask[img_row, img_col, 1] = after
            
        masks[date].append(mask)
    
    ## Make label images
    for date in dates:
        
        Utils.makedir(f"{SAVE_DIR}/{date}")
        
        mask = iter(masks[date])
        b_min = mins[date]
        j_max = maxs[date]
        
        cp.cprint(f"{date} started.\n", "orange")
        length = len(masks[date])
        
        for video_index, (b_min_val, j_max_val) in enumerate(zip(b_min, j_max)):
            
            if (b_min_val < 0):
                continue
            m = next(mask)
            if (video_index < 8):
                continue
            
            for image_index in range(b_min_val, j_max_val+1):
                
                label = np.zeros(MESH, dtype=np.uint8)
                label += m[:, :, 0] < image_index
                label += m[:, :, 1] < image_index
                
                image = np.zeros((*SIZE, 3), dtype=np.uint8)
                for row in range(MESH[0]):
                    for col in range(MESH[1]):
                        image[SIZE[0]//4*row:SIZE[0]//4*(row+1), SIZE[1]//6*col:SIZE[1]//6*(col+1)] += COLORS[label[row, col]]

                pil_image = Image.fromarray(image)
                pil_image.save(f"{SAVE_DIR}/{date}/masked_{video_index+1:02d}_{image_index:04d}.png")
                
                cp.cprint(f"\033[1A{video_index+1} / {length}, {image_index} / {j_max_val}{' '*10}", "orange")
        
        cp.cprint(f"\n\n{date} completed.", "green")
    cp.cprint("\nAll proccess completed !", "green")    
    

def main():
    
    make_after_labeled_image()


main()
