import numpy as np


FRESH_KEYS = ["id", "スランプ", "スランプフロー", "空気量", "N式貫入量", "電気伝導率"]

## 測っていない or わからないものは None
FRESH_VALUES = [
    ["190729_01", 58, 212, 9, 130, 1.59],
    ["190729_02", 41, 202, 6, 97, 1.59],
    ["190729_03", 105, 235, 9, 193, 1.61],
    ["190729_04", 68, 216, 7, 170, 1.74],
    ["190729_05", 141, 259, 8.8, 200, 1.77],
    ["190729_06", 101, 232, 7.6, 200, 1.81],
    ["190731_01", 197, 277, 3.8, 173, 1.85],
    ["190731_02", 176, 264, 4.2, 170, 1.87],
    ["190731_03", 155, 251, 4.5, 167, 1.88],
    ["190731_04", 202, 332, 4.8, 187, 1.67],
    ["190731_05", 195, 305, 4.9, 184, 1.74],
    ["190731_06", 187, 277, 4.9, 180, 1.8],
    ["190731_07", 182, 269, 4.7, 170, 1.83],
    ["190731_08", 154, 251, 4.6, 159, 1.81],
    ["190731_09", 125, 232, 4.5, 147, 1.79],
    ["190731_10", 125, 232, 4.5, 147, 1.79],
    ["190802_01", 182, 277, 4.2, 146, 1.78],
    ["190802_02", 148, 248, 3.9, 148, 1.82],
    ["190802_03", 114, 219, 3.6, 150, 1.86],
    ["190802_04", 103, 231, 5.3, 133, 1.78],
    ["190802_05", 82, 220, 4.5, 122, 1.77],
    ["190802_06", 60, 209, 3.7, 110, 1.76],
    ["190802_07", 120, 225, 6.1, 160, 1.84],
    ["190802_08", 100, 221, 5.5, 138, 1.78],
    ["190802_09", 79, 216, 4.9, 116, 1.71],
    ["190802_10", 79, 216, 4.9, 116, 1.71],
    ["190807_01", 95, 225, 4.6, 163, 1.82],
    ## 190807_02のフレッシュ性状データは測ってない
    ["190807_02", None, None, None, None, None],
    ["190807_03", 75, 210, 3.3, 130, 1.82],
    ["190807_04", 111, 228, 5.1, 153, 1.83],
    ["190807_05", 90, 221, 4.5, 130, 1.85],
    ["190807_06", 68, 213, 3.9, 107, 1.87],
    ["190807_07", 146, 252, 5.1, 170, 1.8],
    ["190807_08", 125, 240, 4.9, 150, 1.84],
    ["190807_09", 103, 227, 4.6, 130, 1.87],
    ["190807_10", 103, 227, 4.6, 130, 1.87],
    ["190809_01", 115, 230, 4.5, 133, 1.83],
    ["190809_02", 102, 225, 4, 137, 1.82],
    ["190809_03", 89, 220, 3.5, 141, 1.8],
    ["190809_04", 180, 283, 4.6, 187, 1.86],
    ["190809_05", 173, 269, 4.6, 159, 1.87],
    ["190809_06", 166, 255, 4.6, 130, 1.87],
    ["190809_07", 120, 230, 5.5, 138, 1.85],
    ["190809_08", 104, 223, 4.8, 129, 1.83],
    ["190809_09", 87, 216, 4, 119, 1.8],
    ["190809_10", 87, 216, 4, 119, 1.8],
    ## 20220714は恐らく実地試験の時のデータ。フレッシュ性状データ？
    ["20220714", None, None, None, None, None],
    ## 20220208、20220209、20220210は、全ての動画のフレッシュ性状データが同じ
    ["20220208", 140, 239, 4.7, 130, 1.75],
    ["20220209", 120, 235, 4.3, 183, 1.75],
    ["20220210", 105, 214, 4, 120, 1.75],
]


def user():
    """
    ユーザー定義の部分
    正規化手法の選択と、保存するCSVのパス
    """
    ## 正規化手法の選択
    func = standardization          ## 標準化
    # func = z_index_normalization  ## 0~1正規化
    
    ## 保存するCSVのパス
    dir = "/Volumes/GoogleDrive/Other computers/iMac/DATA/Laboratory/work"
    file_name = "fresh_data_20220919.csv"
    
    return func, f"{dir}/{file_name}"


def standardization(depthes, key, value):
    """
    標準化(標準正規分布にする)
    """
    ## Noneなら0を出力
    if value is None: return 0
    
    ## y = (x - E[X]) / Sqrt(V[X])
    result = (value - depthes[key]["平均"]) / depthes[key]["標準偏差"] 
    
    return result


def z_index_normalization(depthes, key, value):
    """
    0~1正規化
    """
    ## Noneなら平均を出力
    if value is None: return (depthes[key]["平均"] - depthes[key]["最小値"]) / (depthes[key]["最大値"] - depthes[key]["最小値"])
    
    ## y = (x - Min(X)) / (Max(X) - Min(X))
    result = (value - depthes[key]["最小値"]) / (depthes[key]["最大値"] - depthes[key]["最小値"])
    
    return result
    



## ==================================================================


def calculate_depth(data):
    """
    統計量の計算
    """
    
    ## 最終的に各数値を格納する辞書
    depthes = dict(zip(list(data.keys())[1:], [None]*len(data)))
    
    ## フレッシュ性状データ(変数：data)の数値のみを抽出
    values = [np.array([elem for elem in val if elem is not None], dtype=np.float32) for val in list(data.values())[1:]]
    
    ## 計算
    for key, val in zip(depthes.keys(), values):
        
        depthes[key] = {
            "最小値" : np.min(val),
            "最大値" : np.max(val),
            "平均" : np.average(val),
            "分散" : np.var(val),
            "標準偏差" : np.std(val),
            "中央値" : np.median(val)
        }
    
    return depthes


def write_csv(data, depthes, func, path):
    """
    CSVに書き込み
    """
    
    with open(path, mode="w") as csv:
        keys = list(data.keys())
        csv.write(f"{','.join(keys)}\n")
        for values in zip(*data.values()):
            id, *val = values
            if not "_" in id:
                for place_id in range(10):
                    csv.write(f"{id}_{place_id+1:02d},{','.join(map(str, [func(depthes, keys[i+1], elem) for i, elem in enumerate(val)]))}\n")
            else:
                csv.write(f"{id},{','.join(map(str, [func(depthes, keys[i+1], elem) for i, elem in enumerate(val)]))}\n")


def main():
    
    ## フレッシュ性状データを計算用に整形
    fresh_data = dict(zip(FRESH_KEYS, zip(*FRESH_VALUES)))
    
    ## 統計量の計算
    depthes = calculate_depth(fresh_data)
    
    print("\n".join(["\n| ".join([f"\n{key}"] + [f"{depth}{chr(int('3000', 16))*(max(map(len, depthes.values()))-len(depth))}{elem}" for depth, elem in val.items()]) for key, val in depthes.items()]))
    
    ## 正規化手法とCSVのパスの定義
    func, path = user()
    
    ## CSV書き込み
    write_csv(fresh_data, depthes, func, path)
    
    print("\n完了")
    
    
main()
