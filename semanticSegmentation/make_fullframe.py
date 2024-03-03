import cv2
import numpy as np
import os
import glob
import pickle

from colorPrint import Cprint as cp


DATASET_DIR = "/workspace/Dataset"
SAVE_TEXT = f"/workspace/Dataset/fullframe/text_dataset/all_4class_myself.txt"


class ImageInfo:
    
    with open("/workspace/mesh_dataset/fresh.pickle", mode="rb") as f:
        fresh_data = pickle.load(f)
    with open("/workspace/mesh_dataset/answer.pickle", mode="rb") as f:
        answer_data = pickle.load(f)
    
    def new(self, path:str):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        # self.label = ImageInfo.get_answer(path)
        # if self.label is None: return
        
        path = self.from_str(path)
        self.date = path[2].replace("image_dataset", "")
        self.folder_id = ((path[3].split("_"))[::-1])[0]
        self.place = ((path[4].split("_"))[::-1])[0]
        self.label = path[5]
        self.time = ((path[6].split("_"))[::-1])[1]
        # self.time = ((path[5].split("_"))[::-1])[1]

    
    def from_str(self, path:str):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        lst = path.split("/")
        while (lst[0] == ""):
            lst = lst[1:]
        return lst

    
    @staticmethod
    def from_param(date, folder_id, place, time, label, is_05d:bool=False):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        if is_05d:
            return f"{DATASET_DIR}/image_dataset{date}/CM{date}_{folder_id}/CM{date}_{folder_id}ab_{place:02d}/{label}/CM{date}_{folder_id}ab_{time:05d}_{place:02d}.jpg"
        else:
            return f"{DATASET_DIR}/image_dataset{date}/CM{date}_{folder_id}/CM{date}_{folder_id}ab_{place:02d}/{label}/CM{date}_{folder_id}ab_{time:04d}_{place:02d}.jpg"
        # if is_05d:
            # return f"{DATASET_DIR}/image_dataset{date}/CM{date}_{folder_id}/CM{date}_{folder_id}ab_{place:02d}/CM{date}_{folder_id}ab_{time:05d}_{place:02d}.jpg"
        # else:
            # return f"{DATASET_DIR}/image_dataset{date}/CM{date}_{folder_id}/CM{date}_{folder_id}ab_{place:02d}/CM{date}_{folder_id}ab_{time:04d}_{place:02d}.jpg"
    
    
    @staticmethod
    def get_fresh(path):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        path_elements = path.split("/")
        _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
        # _0, _1, _2, _3, _4, _5, element, *_end = path_elements
        element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
        date, place, time_id, mesh_id = element.split("_")
        key = f"{date}_{place}"
        return " ".join(map(str, ImageInfo.fresh_data[key]))
        
    
    @staticmethod
    def get_answer(path):
        """
        @機能：
        @引数：
        @戻値：
        """
        # /workspace/Dataset/image_dataset190802/CM190802_06/CM190802_06ab_13/CM190802_06ab_1781_13.jpg
        path_elements = path.split("/")
        _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
        # _0, _1, _2, _3, _4, _5, element, *_end = path_elements
        if (element in ["before", "just"]): return
        element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
        
        try:
            date, place, time_id, mesh_id = element.split("_")
        except:
            cp.cprint(f"\033[1AExcept: {path_elements}", "red")
            cp.cprint("reading file...", "green")
            return
        
        try:
            key = f"{date}_{place}_{time_id}_{mesh_id}"
            return ImageInfo.answer_data[key]
        except:
            try:
                key = f"{date}_{place}_0{time_id}_{mesh_id}"
                return ImageInfo.answer_data[key]
            except:
                return

        
    def __str__(self):
        return ", ".join(map(str, [self.date, self.folder_id, self.place, self.label, self.time]))
    

def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


take_paths = lambda name : glob.glob(f"{name[0]}/{name[1]}*")
save_dir = lambda date : f"/workspace/Dataset/fullframe/image/{date}"
mask_dir = lambda date : f"/workspace/Dataset/fullframe/masked_4class_myself/{date}"

without = [f"{DATASET_DIR}/image_dataset190729", f"{DATASET_DIR}/image_dataset210714"]
parent = take_paths([DATASET_DIR, "image"])
info = dict()

cp.cprint("reading file...", "green")

for child in parent:
    if child in without: continue
    folder = take_paths([child, ""])
    for in_folder in folder:        
        place_folder = take_paths([in_folder, ""])
        for label_folder in place_folder:
            before_images = take_paths([label_folder, "before/"])
            just_images = take_paths([label_folder, "just/"])
            for images in before_images+just_images:
            # images_folder = take_paths([label_folder, ""])
            # for images in images_folder:
                img = ImageInfo()
                img.new(images)
                if img.get_answer(images) is None: continue
                if img.date in info.keys():
                    if img.folder_id in info[img.date].keys():
                        if img.place in info[img.date][img.folder_id].keys():
                            if img.time in info[img.date][img.folder_id][img.place].keys():
                                info[img.date][img.folder_id][img.place][img.time].append(img.label)
                            else:
                                info[img.date][img.folder_id][img.place][img.time] = [img.label]
                        else:
                            info[img.date][img.folder_id][img.place] = {img.time : [img.label]}
                    else:
                        info[img.date][img.folder_id] = {img.place : {img.time : [img.label]}}
                else:
                    info[img.date] = {img.folder_id : {img.place : {img.time : [img.label]}}}
                del img

print("\033[1A", end="")

with open(SAVE_TEXT, mode="w"): pass

with open(SAVE_TEXT, mode="a") as f:
    for date in info.keys():
        
        # if (str(date) in ["190802", "190807", "220210", "190809"]): continue
        
        cp.cprint(f"date : {date}    ", "cyan")
        
        for folder_id in info[date].keys():
            
            # if (f"{date}_{folder_id}" in ["190731_07", "190731_01", "190731_03", "190731_06", "190731_08", "190731_10", "190731_04", "190731_09"]): continue
            
            cp.cprint(f"folder_id : {folder_id}", "cyan", bloom=0.8)
        
            frames = [[0]*24 for i in range(int(1e5))]
            for place in info[date][folder_id].keys():
                for time, label in info[date][folder_id][place].items():
                    frames[int(time)][int(place)-1] = label[0]
        
            cp.cprint("none", "gray")
        
            for time, frame_array in enumerate(frames):
                if all(frame_array):
                    frame, mask_frame = np.zeros((270*4, 270*6, 3)), np.zeros((270*4, 270*6, 3))
                    fresh = None
                    for place in range(24):
                        image_path = ImageInfo.from_param(date, folder_id, place+1, time, frame_array[place])
                        if not place:
                            fresh = ImageInfo.get_fresh(image_path)
                        row, col = place//6, place%6
                        try:
                            if (f"{date}_{folder_id}" == "190731_05"):
                                buf = cv2.imread(ImageInfo.from_param(date, folder_id, place+1, time, frame_array[place], is_05d=True))
                            else:
                                buf = cv2.imread(image_path)
                                if buf is None:
                                    buf = cv2.imread(ImageInfo.from_param(date, folder_id, place+1, time, frame_array[place], is_05d=True))
                        except:
                            image_path = ImageInfo.from_param(date, folder_id, place+1, time, frame_array[place], is_05d=True)
                            buf = cv2.imread(image_path)
                        frame[270*row:270*(row+1), 270*col:270*(col+1)] = buf
                        answer = ImageInfo.get_answer(image_path)
                        if answer is None: continue
                        mask_frame[270*row:270*(row+1), 270*col:270*(col+1)] += [0, 64, 128, 255][int(answer)]
                        # if (frame_array[place] == "just"):
                            # mask_frame[270*row:270*(row+1), 270*col:270*(col+1)] += (1<<8)-1
                    # makedir(save_dir(date))
                    makedir(mask_dir(date))
                    img = f"{save_dir(date)}/{folder_id}_{time:04d}.png"
                    msk = f"{mask_dir(date)}/masked_{folder_id}_{time:04d}.png"
                    # cv2.imwrite(img, frame)
                    cv2.imwrite(msk, mask_frame)
                    print(f"{img} {msk} {fresh}", file=f)
                    
                    cp.cprint(f"\033[1A{' '*len('folder_id : ')}{time}", "gray")


cp.cprint("- finished ! -", "green")

