import copy
from functools import partial, lru_cache
import numpy as np
import matplotlib.pyplot as plt


ALLTEXT = "/workspace/mesh_dataset/all.txt"
SAVE_CSV = "/workspace/mesh_dataset/ImagePathTime.csv"
SAVE_PARENT_CSV = "/workspace/mesh_dataset/ImagePathTime_parent.csv"


"""
テキストのBefore-Just間の差が原因かも
"""


class ImagePathTime:
    """
    @機能：ラベル境界からの時間(IPT)の算出
    @関数：ipt(画像のパス)
    """
    
    def __init__(self):
        
        self.IPT = dict()
        self.parent = dict()
        
        with open(SAVE_CSV, mode="r") as f:
            readlines = f.readlines()
        for i, line in enumerate(map(lambda x: x.rstrip("\n"), readlines)):
            if len(line)*i:
                key, before_min, before_max, just_min, just_max = line.split(",")
                self.IPT[key] = [[int(before_min), int(before_max)], [int(just_min), int(just_max)]]
    
        with open(SAVE_PARENT_CSV, mode="r") as f:
            readlines = f.readlines()
        for i, line in enumerate(map(lambda x: x.rstrip("\n"), readlines)):
            if len(line)*i:
                key, before_length, just_length = line.split(",")
                self.parent[key] = [int(before_length), int(just_length)]
    
        self.debug_set = dict()
        
        
    @staticmethod
    def normalization(val, inf, sup, length, ans):
        """
        @機能：正規化
        @引数：
        @戻値：
        """
        
        if ans:
            return (val - inf) / length
        else:
            return (val - sup) / length
    
    
    @staticmethod
    def ipt_color(ipts:list):
        """
        @機能：IPTの値から描画用の色を計算
        @引数：IPTのリスト
        @戻値：
        """
        
        colors = np.array([[255, 0, 0],
                           [0, 0, 255],
                           [0, 255, 0]], dtype=np.float32)
        
        result = [None]*len(ipts)
        for i, ipt in enumerate(ipts):
            if (ipt < 0):
                tmp = colors[0]*(1+ipt) + colors[1]*(-ipt)
            else:
                tmp = colors[2]*ipt + colors[1]*(1-ipt)
            tmp = list(map(lambda x: hex(int(x))[2:], tmp))
            result[i] = "#" + f"{'0'*(2-len(tmp[0]))}{tmp[0]}" + f"{'0'*(2-len(tmp[1]))}{tmp[1]}" + f"{'0'*(2-len(tmp[2]))}{tmp[2]}"
        
        return result
    
    
    def ipt(self, path):
        """
        @機能：ImagePathTimeの算出
        @引数：画像のパス:str
        @戻値：IPT
        """
        
        path_elements = path.split("/")
        _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
        element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
        date, place, time_id, mesh_id = element.split("_")
        key = f"{date}_{place}_{mesh_id}"
        key_parent = f"{date}_{place}"
        value = int(time_id)
        ans = int(ans in ["just", "Just"])
        ## 当該範囲での最小値、最大値
        inf, sup = self.IPT[key][ans]
        length = self.parent[key_parent][ans]
        ipt = ImagePathTime.normalization(value, inf, sup, length, ans)
        ## Beforeの場合、値を負に
        # ipt -= not ans
        
        if not key_parent in self.debug_set.keys():
            self.debug_set[key_parent] = set()
        self.debug_set[key_parent].add(length)
        
        return ipt
    
    
    def make_IPT(self):
        """
        @機能：ImagePathTimeの基準値となる最大値、最小値の計算。(理由なく実行しないで)
        @引数：
        @戻値：
        """
        
        self.IPT = dict()
        self.parent = dict()
        init_value = [[1e5, -1], [1e5, -1]]
        
        print()
        
        with open(ALLTEXT, mode="r") as f:
            readlines = f.readlines()
        length = len(readlines)
        
        for i, line in enumerate(map(lambda x: x.rstrip("\n"), readlines)):
            if len(line):
                path, ans, *_ = line.split(" ")
                # ans = int(ans)
                ans = int("just" in path or "Just" in path)
                
                path_elements = path.split("/")
                _0, _1, _2, _3, _4, _5, _6, element, *_end = path_elements
                element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
                date, place, time_id, mesh_id = element.split("_")
                key = f"{date}_{place}_{mesh_id}"
                # key_parent = f"{date}_{place}"
                value = int(time_id)
                if not key in self.IPT.keys():
                    # self.IPT[key] = copy.deepcopy(init_value)
                    self.IPT[key] = [[1e5, -1], [1e5, -1]]
                # if not key_parent in self.parent:
                    # self.parent[key_parent] = copy.deepcopy(init_value)
                ## DPにより導出
                self.IPT[key][ans][0] = min(self.IPT[key][ans][0], value)
                self.IPT[key][ans][1] = max(self.IPT[key][ans][1], value)
                # self.parent[key_parent][ans][0] = min(self.parent[key_parent][ans][0], value)
                # self.parent[key_parent][ans][1] = max(self.parent[key_parent][ans][1], value)
            print(f"\033[1A{round((i+1)/length * 100, 2)} [%]{' '*50}")
        
        
        for key in self.IPT.keys():
            buf = key.split("_")
            key_parent = "_".join(buf[:2])
            if not key_parent in self.parent.keys():
                self.parent[key_parent] = [-1]*2
            for ans, val in enumerate(self.IPT[key]):
                value = val[1] - val[0]
                self.parent[key_parent][ans] = max(self.parent[key_parent][ans], value)
        
        print(self.parent)
        

    def write_IPT_for_csv(self):
        """
        @機能：CSVへの書き込み
        @引数：
        @戻値：
        """
        
        ## CSV新規作成
        with open(SAVE_CSV, mode="w") as f:
            f.write("key,before_min,before_max,just_min,just_max\n")
        
        ## 書き込み
        with open(SAVE_CSV, mode="a") as f:
            for key, value in self.IPT.items():
                before, just = value
                print(f"{key},{','.join(map(str, before))},{','.join(map(str, just))}", file=f)
                
        ## CSV新規作成
        with open(SAVE_PARENT_CSV, mode="w") as f:
            f.write("key,before_length,before_length\n")
        
        ## 書き込み
        with open(SAVE_PARENT_CSV, mode="a") as f:
            for key, value in self.parent.items():
                before, just = value
                print(f"{key},{before},{just}", file=f)
        
        print("Wrote a csv.")
        
    

def path_maker(date, place, mesh_id, time_id, ans):
    return f"/workspace/Dataset/image_dataset{date}/CM{date}_{place:02d}/CM{date}_{place:02d}ab_{mesh_id:02d}/{ans}/CM{date}_{place:02d}ab_{time_id:04d}_{mesh_id:02d}.jpg"


if(__name__ == "__main__"):
    
    # ipt_ = ImagePathTime()
    # ipt_.make_IPT()
    # ipt_.write_IPT_for_csv()
    
    ipt_ = ImagePathTime()
    
    print(ipt_.ipt(path_maker(date="190731", place=1, mesh_id=23, time_id=1318, ans="before")))
    print(ipt_.ipt(path_maker(date="190731", place=1, mesh_id=23, time_id=1319, ans="before")))
    print(ipt_.ipt(path_maker(date="190731", place=1, mesh_id=23, time_id=1320, ans="just")))
    print(ipt_.ipt(path_maker(date="190731", place=1, mesh_id=23, time_id=1321, ans="just")))
    
    # print(ipt_.ipt("/workspace/Dataset/image_dataset190731/CM190731_01/CM190731_01ab_01/before/CM190731_01ab_0180_01.jpg"))
    # print(ipt_.ipt("/workspace/Dataset/image_dataset190731/CM190731_01/CM190731_01ab_01/before/CM190731_01ab_0200_01.jpg"))
    # print(ipt_.ipt("/workspace/Dataset/image_dataset190731/CM190731_01/CM190731_01ab_01/before/CM190731_01ab_0749_01.jpg"))
    # print(ipt_.ipt("/workspace/Dataset/image_dataset190731/CM190731_01/CM190731_01ab_01/just/CM190731_01ab_0750_01.jpg"))
    # print(ipt_.ipt("/workspace/Dataset/image_dataset190731/CM190731_01/CM190731_01ab_01/just/CM190731_01ab_1000_01.jpg"))
    
    # for i in range(0, 2000):
    #     ans = ["before", "just"][i >= 750]
    #     ipt_num = ipt_.ipt(f"/workspace/Dataset/image_dataset190731/CM190731_01/CM190731_01ab_01/{ans}/CM190731_01ab_{i:04d}_01.jpg")
    #     plt.scatter(i, ipt_num)
    
    # plt.savefig("/workspace/osada_ws/ex_4class/buf.png")
    