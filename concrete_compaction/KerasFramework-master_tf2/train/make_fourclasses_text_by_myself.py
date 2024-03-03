
## my module
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
from line_sender import send_master
from image_path_timer import ImagePathTime as IPT

ipt = IPT()

THRESHOLD = [-0.25, 0.25]

all_text = lambda buf: f"/workspace/mesh_dataset/all_4class{buf}.txt"


def diside_class():
    
    images = []
    answers = []
    freshes = []
    
    with open(all_text(""), mode="r") as f:
        data = f.readlines()
    length = len(data); print()
    for i, line in enumerate(map(lambda x: x.rstrip("\n"), data)):
        buf = line.split(" ")
        if len(buf):
            images.append(buf[0])
            img_ipt = ipt.ipt(buf[0])
            answer_flag = int(img_ipt > 0)
            answer = int(img_ipt >= THRESHOLD[answer_flag]) + 2*answer_flag
            answers.append(int(answer))
            freshes.append(" ".join(buf[2:]))
        print(f"\033[1A{i+1} / {length}")
    with open(all_text("_with_myself"), mode="w"): pass
    with open(all_text("_with_myself"), mode="a") as f:
        for img, ans, frs in zip(images, answers, freshes):
            print(f"{img} {ans} {frs}", file=f)


def main():
    diside_class()


if(__name__ == "__main__"):
    main()