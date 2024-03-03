import pickle

def main():
    
    answer_data = dict()
    
    with open("/workspace/mesh_dataset/all_4class.txt", mode="r") as f:
        lines = f.readlines()
    for line in map(lambda x: x.rstrip("\n"), lines):
        if len(line):
            image, answer, *fresh = line.split(" ")
            path_elements = image.split("/")
            _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
            element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
            date, place, time_id, mesh_id = element.split("_")
            key = f"{date}_{place}_{time_id}_{mesh_id}"
            answer_data[key] = int(answer)
    
    with open("/workspace/mesh_dataset/answer.pickle", mode="wb") as f:
        pickle.dump(answer_data, f)


main()