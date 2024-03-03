# range : [START, END)
START = 150
END = 3117
SPLIT = 1

SIZE = 270
IS_CREATE_FILE = True

DATASET_ID = [807, 1]


text = lambda place : f"/workspace/Dataset/semanticSegmentation/text_dataset/visualize/take_image_{place:02d}.txt"
path = lambda target : f"/workspace/Dataset/image_dataset190{DATASET_ID[0]}/CM190{DATASET_ID[0]}_01/CM190{DATASET_ID[0]}_{DATASET_ID[1]:02d}ab_{target[0]:02d}/CM190{DATASET_ID[0]}_{DATASET_ID[1]:02d}ab_{target[1]:04d}_{target[0]:02d}.jpg"
answer_path = lambda target : f"/workspace/osada_ws/text_dataset/ngc_docker/fold1/{target}.txt"


data_length = (END - START) // SPLIT
indexes = [0]*data_length
indexes[0] = START
for i in range(1, data_length):
    indexes[i] = indexes[i-1] + SPLIT


label = dict()
for target in ["test", "train", "validation"]:
    with open(answer_path(target), mode="r") as f:
        lines = f.readlines()
    for l in lines:
        buffer = l.split(" ")
        while not len(buffer[0]):
            buffer = buffer[1:]
        if str(f"CM190{DATASET_ID[0]}_01") in buffer[0]:
            label[buffer[0].replace("//", "/").replace("/just", "").replace("/before", "")] = int(buffer[1])
for place in range(24):
    default_label = 0
    if IS_CREATE_FILE:
        with open(text(place+1), mode="w") as f: pass
    with open(text(place+1), mode="a") as f:
        for index in indexes:
            try:
                default_label = label[path([place+1, index])]
                print(f"{path([place+1, index])} {default_label}", file=f)
            except:
                print(f"{path([place+1, index])} {default_label}", file=f)
