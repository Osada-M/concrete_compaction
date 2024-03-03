import pickle


DIR = "/workspace/osada_ws/text_dataset/ngc_docker"


def make_fold_dict():
    """
    @機能：
    @引数：
    @戻値：
    """
    
    fold_data = dict()
    
    for fold in range(1, 6):
        for target in ["train", "validation", "test"]:
            with open(f"{DIR}/fold{fold}/{target}.txt", mode="r") as f:
                lines = f.readlines()
            for i, line in enumerate(map(lambda x: x.rstrip("\n"), lines)):
                if len(line):
                    image, answer, *fresh = line.split(" ")
                    image = image.replace("//", "/")
                    path_elements = image.split("/")
                    _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
                    element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
                    date, place, time_id, mesh_id = element.split("_")
                    key = f"{date}_{place}"
                    
                    if not (key in fold_data.keys()):
                        fold_data[key] = [None]*5
                    fold_data[key][fold-1] = target
    
    fold_data = dict(sorted(fold_data.items()))

    with open(f"{DIR}/fold.pickle", mode="wb") as f:
        pickle.dump(fold_data, f)

    print(fold_data)
    

def split_fold():
    """
    @機能：
    @引数：
    @戻値：
    """
    
    with open(f"{DIR}/fold.pickle", mode="rb") as f:
        fold_data = pickle.load(f)
    
    ## 書き込むテキストを新規作成
    # for fold in range(1, 6):
    #     for target in ["train", "validation", "test"]:
    #         with open(f"{DIR}/fold{fold}/{target}_4class.txt", mode="w"): pass
    
    ## FOLDを振り分けて書き込み
    # with open(f"{DIR}/all_4class.txt", mode="r") as f:
    #     lines = f.readlines()
    # for i, line in enumerate(map(lambda x: x.rstrip("\n"), lines)):
    #     if len(line):
    #         image, answer, *fresh = line.split(" ")
    #         path_elements = image.split("/")
    #         _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
    #         element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
    #         date, place, time_id, mesh_id = element.split("_")
    #         key = f"{date}_{place}"
    #         fold_dict = fold_data[key]
            
    #         for fold in range(1, 6):
    #             with open(f"{DIR}/fold{fold}/{fold_dict[fold-1]}_4class.txt", mode="a") as f:
    #                 print(line, file=f)
    
    ## 訓練データだけ５個飛ばし
    for fold in range(1, 6):
        with open(f"{DIR}/fold{fold}/train_4class.txt", mode="r") as f:
            lines = f.readlines()[::5]
        with open(f"{DIR}/fold{fold}/train_4class.txt", mode="w") as f:
            f.write("".join(lines))
    
    
    print("finished")


def main():
    make_fold_dict()
    split_fold()


main()
