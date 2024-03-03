"""
@file   visualization_predicted_result.py
@brief  predict_combined_list.csvをもとにグラフを作り直す(軸ラベルあり, 余白の透過).
        190731-190809の動画以外(?)のグラフ作成もできる(エラーを回避した).
        各メッシュのbefore/justの予測値を可視化する.
@note   原因はday_num_video_list()のvideo_num(動画ファイル名)のスライス.
        01abの場合は[:-2]
        01aの場合は[:-1]
        01の場合はスライス無し.
"""

from tqdm import tqdm
from os import makedirs

import argparse
import yaml
import csv
import re
import pathlib
import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import classification_report

def cleate_parser_and_fetch_yaml():
    """
    @fn     cleate_parser_and_fetch_yaml()
    @brief  アーグパーサの生成とyamlの読み込み, yamlオブジェクトを返す
    @return yamlオブジェクト
    """
    # パーサーの生成
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', help='config yaml file')

    # コマンド引数の受取
    args = parser.parse_args()
    print('config file:', args.conf)

    # コマンド引数(yamlファイル)の読み込み
    with open(args.conf) as oepn_file:
        print('complete!\n')
        yml = yaml.safe_load(oepn_file)
        
    return yml

def extract_video_num(row):
    """
    @fn         extract_video_num()
    @brief      画像ファイル名を抽出
    @param[row] predicted_combined_listの1行分
    """
    image_name = row[1].split('/')[-1]
    image_name = image_name.split('.')[0]
    video_num = image_name.split('_')[-3]
    video_num = video_num[:2]      #名前を分割後、abをなくす
    return video_num

def day_num_video_list(row):
    """
    @fn         day_num_video_list()
    @brief      csvの1行から動画ファイル名を取得(撮影日, 撮影番号: CM190729_07とか)
    @param[row] predicted_combined_listの1行分
    @note       動作例を書いていく.
                row: ['0', '/workspace/Dataset/image_dataset210714/CM210714_02/CM210714_02_01/before/CM210714_02_2038_01.jpg', '0', '0', '"[[0.9783307313919067', ' 0.021669279783964157]]"']
                image_name(1): 'CM210714_02_2038_01.jpg'
                image_name(2): 'CM210714_02_2038_01'
                video_num(1): '02'
                video_num(2): '02'
                day_num: 'CM210714'
                day_num_video: 'CM210714_02'
    """
    image_name = row[1].split('/')[-1]
    image_name = image_name.split('.')[0]
    video_num = image_name.split('_')[-3]
    video_num = video_num[:-2]      #名前を分割後、abをなくす
    day_num = image_name.split('_')[0]
    day_num_video = day_num + '_' + video_num     #動画の日付と番号をまとめたもの

    return day_num_video

def create_graph(reader):
    fig, ax = plt.subplots()

    # ?
    correct_label = [[] for i in range(24)]
    
    #正解ラベルと実験結果が一致しているかしていないか
    pred_result = [[] for i in range(24)]
    mesh_index_list = []

    # csvを1行づつ読んでいく.
    for row in reader:
        # 画像ファイル名を抽出. ex) /media/nagalab/.../CM210714_01_01_0512.jpg => CM210714_01_01_0512.jpg
        image_name = row[1].split('/')[-1]
        # image_nameから拡張子を除く. ex) CM210714_01_01_0512.jpg => CM210714_01_01_0512
        image_name = image_name.split('.')[0]
        video_num = image_name.split('_')[-1]

        mesh_num = row[1].split('/')[-1]
        mesh_num = mesh_num.split('.')[0]

        mesh_index = mesh_num.split('_')[-2]
        mesh_index_list.append(mesh_index)
    
        if video_num == '01':
            len_index_list = int(len(mesh_index_list))

        sample_day = image_name.split('_')[0]
        sample_num = image_name.split('_')[1]

        mesh_index = int(video_num)-1  

        #for i in row:
        if row[2] == row[3]:
            pred_result[mesh_index].append('true')
        else:
            pred_result[mesh_index].append('false')

        if row[2] == '0':
            correct_label[mesh_index].append('before')
            
            #print(counter)
        else:
            correct_label[mesh_index].append('just')
        #print(correct_label)
        #print(np.array(correct_label).shape)

    for i in range(24):
        index_before = [i for i, x in enumerate(correct_label[i]) if x == 'before']
        index_before = list(zip(index_before,np.ones(len(index_before))))
        #print(index_before)
        index_just = [i for i, x in enumerate(correct_label[i]) if x == 'just']
        index_just = list(zip(index_just,np.ones(len(index_just))))
        #print(index_just)
        index_true = [i for i, x in enumerate(pred_result[i]) if x == 'true']
        index_true = list(zip(index_true,np.ones(len(index_true))))
        #print(index_true)
        index_false = [i for i, x in enumerate(pred_result[i]) if x == 'false']
        index_false = list(zip(index_false,np.ones(len(index_false))))
        #print(index_false)
        #index_mesh = [i for i, x in enumerate(mesh_index[i]) if x == 'empty']
        #index_mesh = list(zip(index_mesh,np.ones(len(index_mesh))))

        yrange1 = [i+1, 0.5]
        yrange2 = [i+0.5, 0.5] 

        ax.broken_barh(xranges=index_false, yrange=yrange2, facecolor='black')
        ax.broken_barh(xranges=index_true,yrange=yrange2, facecolor='white')
        ax.broken_barh(xranges=index_before, yrange=yrange1, facecolor='red')
        ax.broken_barh(xranges=index_just,yrange=yrange1, facecolor='green')
        #ax.broken_barh(xranges=index_just,yrange=yrange1, facecolor='purple')
        ax.set_yticks(list(range(1, 25)))
        # ax.set_xticks()
        ax.set_xlim([0, len_index_list])
        # ax.set_ylim([0, 25])

    ax.set_xlabel('frame(30fps)')
    ax.set_ylabel('mesh number')

    # plt.savefig(yml["TESTModel"]["path"] + sample_day +"_"+ sample_num+ "_result.png")
    fig.patch.set_alpha(0)
    plt.savefig('./result/2020_{}_{}_result.png'.format(sample_day, sample_num))
    print('len_index_list:', len_index_list)
    plt.title(sample_day + "_" + sample_num)


def create_predicted_figs(csv_path):
    """
    @fn     create_predicted_figs()
    @brief  predict_combined_list.csvを元に, テストグラフを作成する.
    """
    # yml = cleate_parser_and_fetch_yaml()

    mesh_list = []

    # ?
    local_index = 0
    # ?
    i = 0
    with open(csv_path) as f:
        """
        どのビデオを使っているかを取得する.
        ex) CM210714_02, cM210714_07の2本のビデオを使ってるみたいな.
        """
        reader = csv.reader(f)
        header = next(reader)

        for row in tqdm(reader):
            # '/workspace/Dataset/.../before/CM210714_02_2038_01.jpg'から'cM210714_02を抽出
            video_num = day_num_video_list(row)
            
            # mesh_listに追加.
            mesh_list.append(video_num)
            # mesh_listの重複を無くす
            mesh_set = set(mesh_list)
            # mesh_setをリストにする
            mesh_set_list = list(mesh_set) 
            # mesh_set_listをソート(昇順).
            sort_mesh_set_list = sorted(mesh_set_list)
        
        print('sort_mesh_set_list:', sort_mesh_set_list)
        # exit()
        """
        画像データごとにcsvファイルを分割
        """
        f.seek(0)
        reader = csv.reader(f)
        header = next(reader)

        #split_resultをmesh_set_resultの長さだけのリストにする 例）長さが２の場合[0,1]
        split_result = []   
        split_result = [[] for _ in range(len(sort_mesh_set_list))]
        for row in tqdm(reader):
            # 読み込んだ行の動画番号を抽出
            video_num2 = extract_video_num(row)     

            if video_num2 != local_index:
                i += 1
                local_index = video_num2
            split_result[i-1].append(row)
        for i in range(len(sort_mesh_set_list)):
            create_graph(split_result[i])


def make_dir_of_test_videos(ex_dir_path, isMake=True) -> list:
    """
    @fn             make_dir_of_test_videos
    @brief          テストに使われる動画のディレクトリを実験ディレクトリに作成, 動画リストを返す
    @param[yml]     実験ディレクトリ
    @param[isMake]  ディレクトリを作成するか否か(default: True).
    @return         テストに使われる動画名(CM190729_01とか)のリスト
    """

    test_video_list = list()
    
    print('Check the video used for the test')
    with open('%stest_predict_combined_list.csv'%ex_dir_path) as prd_csv:
        csv_reader = csv.reader(prd_csv)
        next(csv_reader)

        for line in tqdm(csv_reader):
            shooting_day_and_video_number = day_num_video_list(line)
            if shooting_day_and_video_number in test_video_list:
                pass
            else:
                test_video_list.append(shooting_day_and_video_number)
    
    print('test_video_list:', test_video_list)

    # 動画毎のディレクトリを作成.
    if isMake:
        for videos in test_video_list:
            makedirs('%s%s'%(ex_dir_path, videos), exist_ok=True)
    
    return test_video_list


def extract_just_predict_value(prd) -> float:
    """
    @fn         extract_just_predict_value()
    @brief      予測値('[[0.01, 0.99]]')からjustの確信度(0.99)を抽出
    @param[prd] 予測値の文字列('[[0.01, 0.99]]')
    @return     justの予測値
    """

    # カンマで分割し, 最後の要素を抽出. => '[[0.01, 0.99]]' -> ['[[0.01', '0.99]]'] -> '0.99]]'
    split_comma = prd.split(',')[-1]
    # 数字のみ抽出. ([〜]: 〜内のいずれか1文字, +: 直前のパターンの1回以上の繰り返し.)
    str_just_prd_value = re.search('[0-9.e-]+', split_comma).group()

    return float(str_just_prd_value)


def extract_mesh_number(img_path) -> str:
    """
    @fn                 extract_mesh_number()
    @brief              画像パス名から, メッシュ番号を抽出する.
    @param[img_path]    画像パス
    @return             メッシュ番号(文字列)
    """

    # スラッシュで分割し, 画像名のみを抽出
    split_str = img_path.split('/')[-1]
    # 拡張子を省く
    split_str = split_str.split('.')[0]
    # メッシュ番号を抽出
    mesh_num = split_str.split('_')[-1]

    return mesh_num


def create_plt_object():
    """
    @fn     create_plt_fig()
    @brief  pltのオブジェクトを作成
    @return 戻り値無し
    """
    plt.figure(figsize=(9, 5))
    # y軸小数点以下2桁表示
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    return


def generate_plt_fig(target_dict: dict, isShow=False, isSave=True, isClose=False):
    """
    @fn                 generate_plt_fig()
    @brief              グラフを保存する関数
    @param[target_dict] 保存先とかメッシュ番号とかの辞書
    @param[isShow]      表示するか否か.
    @param[isSave]      グラフを保存するか否か.
    @param[isClose]     保存毎にメモリを開放するか否か.
    @return             戻り値無し
    """
    # グラフ要素の設定
    plt.title('Predicted value of just in mesh {} of video {}'.format(target_dict['current_mesh'], target_dict['current_video_name']))
    plt.xlabel('frame(30fps)')
    plt.ylabel('predicted value')
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.grid(axis='y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    
    if isShow:
        plt.show()
    
    if isSave:
        plt.savefig(
            '{0}{1}/vs_{2}_{1}_{3}.png'.format(target_dict['ex_dir_path'], target_dict['current_video_name'], target_dict['base_ex_name'], target_dict['current_mesh']),
            bbox_inches="tight",
            pad_inches=1.0
        )
    
    if isClose:
        plt.close()
    
    return
    

def extract_values_from_csv(csv_path: str) -> dict:
    """
    @fn                 extract_values_from_csv()
    @brief              csvファイルからカラムをkey, 各値をvalueのリストとしてdictを作成
    @param[csv_path]    csvファイルのパス
    @return             just_prds, answersの辞書
    """
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        prd_values, answers = [], []
        for row in csv_reader:
            prd_values.append(float(row[1]))
            answers.append(int(row[2]))
        
    return {'prds': prd_values, 'anss': answers}


def compare_two_predict_values(target_prefix: str, base_prefix: str, target_ex_name='target', base_ex_name='base', isShow=False):
    """
    @fn                     compare_two_predict_values()
    @brief                  2つのテスト結果より, justの確信度を比較するグラフを作成する.
    @param[target_prefix]   比較対象の実験ディレクトリ(+ Metricとか).
    @param[base_prefix]     比較基準の実験ディレクトリ(DiffNormとか).
    @param[target_ex_name]  比較対象の実験名(default: target).
    @param[base_ex_name]    比較基準の実験名(default: base).
    @param[isShow]          作成されるグラフを表示するか否か(default: False).
    """

    # テスト動画名を取得.
    video_list = make_dir_of_test_videos(target_prefix, isMake=False)
    
    # 各ディレクトリの位置のpathlibのオブジェクトを作成
    target_dir = pathlib.Path(target_prefix)
    base_dir = pathlib.Path(base_prefix)

    print('\nStart creating the graph')
    # 各動画ごとのディレクトリに入る
    for video_name in tqdm(video_list):
        target_each_video_dir = target_dir.joinpath(video_name)
        base_each_video_dir = base_dir.joinpath(video_name)

        # video_nameディレクトリ内にある, csvファイルを取得
        target_each_csvs = sorted(list(target_each_video_dir.glob('*.csv')))
        base_each_csvs = sorted(list(base_each_video_dir.glob('*.csv')))

        # print('target_each_csvs:', target_each_csvs)
        # print('base_each_csvs:', base_each_csvs)
        # exit()

        # print(target_each_csvs)
        # exit()

        for target_csv_file, base_csv_file in zip(target_each_csvs, base_each_csvs):
            target_values_dict = extract_values_from_csv(target_csv_file)
            base_values_dict = extract_values_from_csv(base_csv_file)

            create_plt_object()
            plt.plot(target_values_dict['prds'], color='darkcyan', label=target_ex_name, zorder=3)
            plt.plot(base_values_dict['prds'], color='darkgray', label=base_ex_name, linewidth=1, zorder=2)
            plt.plot(target_values_dict['anss'], color='black', label='answer', zorder=1)

            param = {
                'ex_dir_path'       : target_prefix,
                'current_video_name': video_name,
                'current_mesh'      : str(target_csv_file).split('/')[-1].split('.csv')[0].split('_')[-1],
                'base_ex_name'      : base_ex_name
            }
            generate_plt_fig(param, isShow=isShow, isSave=True, isClose=True)
    plt.clf()


def create_predicted_value_of_each_mesh(yml, isShow=False, isSave=True, isExpt=True):
    """
    @fn             create_predicted_value_of_each_mesh()
    @brief          各メッシュのjustの予測値をcsvから抽出し, 可視化する.
    @param[yml]     yamlオブジェクト
    @param[isShow]  作成したグラフを表示するか否か(default: False).
    @param[isSave]  pngとして画像を保存するか否か(default: True).
    @param[isExpt]  csvとしてjust確信度を保存するか否か(default: True).
    @return         戻り値無し.
    """
    
    # 実験のディレクトリを取得(.../trainからの相対パス)
    ex_dir_path = yml['TESTModel']['path']
    # テスト動画名(CM190729_01abとかCM190805_04abとかのリスト)を取得, ディレクトリの作成.
    test_video_list = make_dir_of_test_videos(yml['TESTModel']['path'])

    # csvファイルを開く
    with open('%stest_predict_combined_list.csv'%ex_dir_path) as predicted_csv:
        # 行数を取得(終端-1と, ヘッダ分で-2)
        max_row = len(predicted_csv.readlines()) - 2
        
        predicted_csv.seek(0)
        # csvオブジェクトの取得と, ヘッダーの読み飛ばし
        csv_reader = csv.reader(predicted_csv)
        header = next(csv_reader)

        # current: 操作対象のこと. for文で扱っているのがCM190729(current_video_num), 01(current_mesh)ってことを記憶するため.
        current_mesh = '01'
        current_video_name = test_video_list[0]

        # justの予測値郡を格納
        just_prds = list()
        # 正解ラベル郡を格納
        answers = list()

        # 1行ずつ読んでいく
        for cnt_row, row in tqdm(enumerate(csv_reader)):
            img_path = row[1]
            answer = row[2]
            predicts = row[4]

            # メッシュ番号の取得
            mesh_num_str = extract_mesh_number(img_path)
            # justの確信度を取得
            just_prd_val = extract_just_predict_value(predicts)
            # 動画名の取得
            video_name = day_num_video_list(row)

            # 最後の行だったとき, 次のイテレートがないので24以外の何かを入れとく.
            if cnt_row == max_row:
                mesh_num_str = 'fin'

            # 操作対象のメッシュか否か?
            if current_mesh == mesh_num_str:
                # 予測値と正解を格納
                just_prds.append(just_prd_val)    
                answers.append(answer)
            # メッシュが変わったらグラフを作る
            else:
                # グラフを作る.
                create_plt_object()

                # プロット
                plt.plot(just_prds, label='predicted value of just', zorder=2)
                plt.plot(answers, color='black', label='answer', zorder=1)

                # グラフ要素の設定
                plt.title('Predicted value of just in mesh {} of video {}'.format(current_mesh, video_name))
                plt.xlabel('frame(30fps)')
                plt.ylabel('predicted value')
                plt.yticks([0, 0.25, 0.5, 0.75, 1])
                plt.grid(axis='y')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
                
                if isShow:
                    plt.show()
                
                if isSave:
                    # グラフの保存
                    plt.savefig('{0}{1}/{1}_{2}.png'.format(ex_dir_path, current_video_name, current_mesh), bbox_inches="tight", pad_inches=1.0)

                if isExpt:
                    # 予測値, 正解ラベルの作成と保存
                    df = pd.DataFrame({'just_prds': just_prds, 'answers': answers})
                    df.to_csv('{0}{1}/{1}_{2}.csv'.format(ex_dir_path, current_video_name, current_mesh))

                # pltの削除
                plt.close()
                
                # 作業対象メッシュ, 動画名を更新
                current_mesh = mesh_num_str
                current_video_name = video_name

                # リストの初期化
                just_prds = list()
                answers = list()
    
    # 開放
    plt.clf()
    print('='*16 + ' FINISH ' + '='*16)
    return

def make_classification_report_with_predict_combined_list(prd_cmb_path):
    """
    @fn                     make_classification_report_with_predict_combined_list()
    @brief                  動画ごとにclassification_reportを作る.
    @param[prd_cmb_path]    predict_combined_list.cscのパス.
    """

    with open(prd_cmb_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        # header = next(csv_reader)
        
        prd_list = list()
        ans_list = list()

        for row in tqdm(csv_reader):
            shooting_day_and_number = row[1].split('/')[4]
        

            ans_list.append(int(row[2]))
            prd_list.append(int(row[3]))

    # prd_list, ans_listが空じゃないときにclassification_reportを作成
    result = classification_report(ans_list, prd_list, target_names=['before', 'just'], output_dict=True)
    classification_dataframe = pd.DataFrame(result).T
    classification_dataframe.to_csv('./result/%s_classification_report.csv'%shooting_day_and_number)
    classification_dataframe.to_excel('./result/test_DemExperiment/%s_classification_report.xlsx'%shooting_day_and_number)

    print(classification_dataframe)

    return
        

if __name__ == '__main__':
    config_yaml = cleate_parser_and_fetch_yaml()

    # justの確信度をメッシュ毎に確認する
    # create_predicted_value_of_each_mesh(config_yaml, isShow=False, isSave=False, isExpt=True)
    
    fold = config_yaml['Resourcedata']['resourcepath'].split('/')[-2][-1]
    # 実験2つを比較.
    compare_two_predict_values(
        target_prefix=config_yaml['TESTModel']['path'],
        base_prefix='./result/gray_resnet18_lr5_f%s/'%(fold),
        target_ex_name='ResNet18 + BC_Learning',
        base_ex_name='ResNet18',
        isShow=False
    )
    # compare_two_predict_values(
    #     target_prefix=config_yaml['TESTModel']['path'],
    #     base_prefix='./result/NiN_Comparison/NiN_Metric_BC/mlt_BC_kullback_NiN_SphereFace_lr5_f3/',
    #     target_ex_name='useBC',
    #     base_ex_name='Mix_label',
    #     isShow=False
    # )

    # create_predicted_figs('/media/nagalab/Volume01/kojima_ws/concrete_compaction/KerasFramework-master_tf2/train/predict_combined_list.csv')
    # create_predicted_figs('/media/nagalab/SSD1.7TB/nagalab/kojima_ws/concrete_compaction/KerasFramework-master_tf2/train/result/ABST_EX_DONT_USE_FRESH/predict_combined_list.csv')
    # make_classification_report_with_predict_combined_list('/media/nagalab/SSD2.0TB/kojima_ws/concrete_compaction/KerasFramework-master_tf2/train/result/test_DemExperiment/210714_01_prd.csv')
    # make_classification_report_with_predict_combined_list('/media/nagalab/SSD2.0TB/kojima_ws/concrete_compaction/KerasFramework-master_tf2/train/result/test_DemExperiment/210714_02_prd.csv')
    # make_classification_report_with_predict_combined_list('/media/nagalab/SSD2.0TB/kojima_ws/concrete_compaction/KerasFramework-master_tf2/train/result/test_DemExperiment/210714_03_prd.csv')
    # make_classification_report_with_predict_combined_list('/media/nagalab/SSD2.0TB/kojima_ws/concrete_compaction/KerasFramework-master_tf2/train/result/test_DemExperiment/210714_04_prd.csv')
    # make_classification_report_with_predict_combined_list('/media/nagalab/SSD2.0TB/kojima_ws/concrete_compaction/KerasFramework-master_tf2/train/result/test_DemExperiment/210714_05_prd.csv')
    # make_classification_report_with_predict_combined_list('/media/nagalab/SSD2.0TB/kojima_ws/concrete_compaction/KerasFramework-master_tf2/train/result/test_DemExperiment/210714_06_prd.csv')
    # make_classification_report_with_predict_combined_list('/media/nagalab/SSD2.0TB/kojima_ws/concrete_compaction/KerasFramework-master_tf2/train/result/test_DemExperiment/210714_07_prd.csv')
    # make_classification_report_with_predict_combined_list('/media/nagalab/SSD2.0TB/kojima_ws/concrete_compaction/KerasFramework-master_tf2/train/result/test_DemExperiment/210714_08_prd.csv')
    # make_classification_report_with_predict_combined_list('/media/nagalab/SSD2.0TB/kojima_ws/concrete_compaction/KerasFramework-master_tf2/train/result/test_DemExperiment/210714_09_prd.csv')
    # make_classification_report_with_predict_combined_list('/media/nagalab/SSD2.0TB/kojima_ws/concrete_compaction/KerasFramework-master_tf2/train/result/test_DemExperiment/210714_10_prd.csv')