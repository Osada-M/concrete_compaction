from colorPrint import Cprint as cp


ALL_PARHS_TEXT = "/media/nagalab/SSD1.7TB/nagalab/Dataset/fullframe/text_dataset/all.txt"
OUTPUT_DIR = "/media/nagalab/SSD1.7TB/nagalab/Dataset/fullframe/text_dataset"
# ALL_PARHS_TEXT = "/workspace/mesh_dataset/all_4class.txt"
# ALL_PARHS_TEXT = "/media/nagalab/SSD1.7TB/nagalab/Dataset/semanticSegmentation/text_dataset/all_4class.txt"
# OUTPUT_DIR = "/media/nagalab/SSD1.7TB/nagalab/Dataset/semanticSegmentation/text_dataset"
SPLIT_MATRIX = [0, 1, 2, 2, 2, 0, 1, 2, 2, 2]
SPLIT_KEY = ["test", "validation", "train"]

IS_TRAIN_DATA_SKIP = True
SKIP_NUMBER = 5
IS_DOCKER = True

text_path = lambda target : f"{OUTPUT_DIR}/fold{target[0]}/{target[1]}_rectified.txt"


if IS_DOCKER:
    ALL_PARHS_TEXT = ALL_PARHS_TEXT.replace("media/nagalab/SSD1.7TB/nagalab", "workspace")
    OUTPUT_DIR = OUTPUT_DIR.replace("media/nagalab/SSD1.7TB/nagalab", "workspace")


def right_rotate(array:list):
    return [array[-1]] + array[:-1]


def adjust_matrix(fold:int):
    matrix = SPLIT_MATRIX.copy()
    for i in range(fold):
        matrix = right_rotate(matrix)
    
    return matrix


def extract_paths():
    with open(ALL_PARHS_TEXT, "r") as f:
        lines = f.readlines()
    
    return lines


def make_dataset():
    lines = extract_paths()
    for fold in range(5):
        matrix = adjust_matrix(fold)
        fold += 1
        count = -1
        for l in map(lambda x: x.rstrip("\n"), lines):
            buffer = l.split(" ")
            image, masked, *fresh = buffer
            # image = buffer[0]
            # masked = buffer[1].rstrip("\n")
            image_index = int((image.replace(".jpg", "").split("/")[-1]).split("_")[0])-1
            image_split = SPLIT_KEY[matrix[image_index]]
            count += image_split == "train"
            if (image_split == "train") and IS_TRAIN_DATA_SKIP and (count%SKIP_NUMBER): continue
            with open(text_path([fold, image_split]), "a") as f:
                print(f"{image} {masked} {' '.join(fresh)}", file=f)
        cp.cprint(f"completed : fold {fold}", "green")
        

def make_text():
    for fold in range(5):
        for target in SPLIT_KEY:
            with open(text_path([fold+1, target]), "w") as f: pass


def main():
    make_text()
    make_dataset()


if(__name__ == "__main__"):
    main()