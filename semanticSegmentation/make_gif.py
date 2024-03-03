from PIL import Image
import cv2
import numpy as np

from colorPrint import Cprint as cp


## ================ config ===================


DURATION = 100.
LOAD_ID = "unet_20220222_prefullframe_finetuning_b2_e10_fold4_80-75"
LOAD_DIR = f"/workspace/semanticSegmentation/visualize/{LOAD_ID}"
SAVE_DIR = f"/workspace/semanticSegmentation/visualize/{LOAD_ID}"
SAVE_ID = f"CM190807_{int(DURATION)}ms"
# RANGE = [50, 250]
RANGE = [0, 195]
SIZE = 270
IS_MAKE_GIF = True
# IS_MAKE_MP4 = False
IS_SAVE_FULLFRAME = False

IS_FULLFRAME = True
# SIZE_COEF = [1, 1]


## ===========================================


length = RANGE[1] - RANGE[0]
images = [None]*length
# if IS_MAKE_MP4:
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     video = cv2.VideoWriter(f"{SAVE_DIR}/{SAVE_ID}.mp4", fourcc, 1000./DURATION, (SIZE*6, SIZE*4), isColor=False)

print()

for time in range(length):
    if IS_FULLFRAME:
        # frame = np.zeros((SIZE*SIZE_COEF[0], SIZE*SIZE_COEF[1], 3))
        frame = cv2.imread(f"{LOAD_DIR}/{time+RANGE[0]:04d}.png")
    else:
        frame = np.zeros((SIZE*4, SIZE*6, 3))
        for place in range(24):
            row, col = place//6, place%6
            frame[SIZE*row:SIZE*(row+1), SIZE*col:SIZE*(col+1)] = cv2.imread(f"{LOAD_DIR}/{place+1:02d}/{time+RANGE[0]:04d}.png")
    if IS_MAKE_GIF:
        images[time] = Image.fromarray(cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2RGB).astype(np.uint8))
        if IS_FULLFRAME:
            images[time] = images[time].resize([SIZE*3, SIZE*2])
    # if IS_MAKE_MP4:
        # cv2.imwrite(f"{SAVE_DIR}/buffer.png", frame)
        # video.write(cv2.imread(f"{SAVE_DIR}/buffer.png"))
    if IS_SAVE_FULLFRAME:
        cv2.imwrite(f"{SAVE_DIR}/fullframe/{time+RANGE[0]:04d}.png", frame)
    cp.cprint(f"\033[1Acompleted : {time+1} / {length}", "green")

if IS_MAKE_GIF:
    images[0].save(f"{SAVE_DIR}/{SAVE_ID}.gif", save_all=True, append_images=images, duration=DURATION)
# if IS_MAKE_MP4:
    # video.release()

cp.cprint("- finished ! -", "cyan")
