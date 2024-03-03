import random

from colorPrint import Cprint as cp


RANDOM_SEED = 1
FOLD_MATRIX = [[0, 2, 1, 1, 1],
               [1, 0, 2, 1, 1],
               [1, 1, 0, 2, 1],
               [1, 1, 1, 0, 2],
               [2, 1, 1, 1, 0]]


work_path = lambda target : f"/workspace/Dataset/fullframe/text_dataset/{target}"
keys = ["test", "train", "validation"]

with open(work_path("all.txt"), mode='r') as f:
    lines = f.readlines()
data = []
for l in lines:
    buffer = l.split(" ")
    data.append([buffer[0], buffer[1].rstrip("\n")])

random.seed(RANDOM_SEED)
random.shuffle(data)
datacount = len(data)
split_number = datacount//5
data_accums = [split_number]*4
data_accums += [datacount-(4*split_number)]

index = 0
data_fold = [[] for _ in range(5)]
for i, number in enumerate(data_accums):
    for _ in range(number):
        data_fold[i].append(data[index])
        index += 1

output = dict(zip(range(5), [dict(zip(keys, [[] for _ in range(3)])) for __ in range(5)]))
for fold in range(5):
    fold_split = FOLD_MATRIX[fold]
    for i, fd in enumerate(fold_split):
        output[fold][keys[fd]] += data_fold[i]
    for key in keys:
        with open(work_path(f"fold{fold+1}/{key}.txt"), mode="w") as f: pass
        with open(work_path(f"fold{fold+1}/{key}.txt"), mode="a") as f:
            without_count = -1
            count = 0
            for img, msk in output[fold][key]:
                if (key == "train"):
                    without_count += 1
                    if without_count%5: continue
                print(f"{img} {msk}", file=f)
                count += 1
            cp.cprint(f"fold : {fold}, target : {key}, length : {count} - {len(output[fold][key])}", "cyan")
            