TEXT_DATASET = "/media/nagalab/SSD1.7TB/nagalab/kojima_ws/concrete_compaction/text_dataset/ngc_docker"


text = lambda *target: f"{TEXT_DATASET}/fold{target[0]}/{target[1]}.txt"


def take_path():
    images = [None]*5
    for fold in range(5):
        images[fold] = dict()
        for target in ["train", "test", "validation"]:
            images[fold][target] = []
            with open(text(fold+1, target)) as f:
                lines = f.readlines()
            for l in lines:
                images[fold][target].append(l.split(" ")[0])
    
    return images


def extract_detail(images:list):
    detail = [None]*5
    for i, fold in enumerate(images):
        detail[i] = dict()
        for target, val in fold.items():
            detail[i][target] = dict()
            for path in val:
                buffer = path.split("/").pop()
                buffer = (buffer.replace(".jpg", "")).split("_")
                key = f"{buffer[0]}_{buffer[1]}"
                detail[i][target][key] = detail[i][target][key]+1 if key in detail[i][target] else 1
    
    return detail


def main():
    images = take_path()
    detail = extract_detail(images)
    for i, fold in enumerate(detail):
        print(f"== {i} ==")
        for target, val in fold.items():
            print(target)
            print(val)
        print()


if(__name__ == "__main__"):
    main()
