# concrete_reiwa4
令和4年度(2022/02 ~ 2023/04)のコンクリ。長田(おさだ)。


<hr/>

## 新たなクラスの分類区分

左）[ResNet18] -> [UMAP]で得た図。赤〜青:Before、青〜緑:Just<br>
右）左図にk-meansをかけて4つにクラスタリングした図<br>
<img src="https://user-images.githubusercontent.com/76993392/180642301-afe80fdf-2d89-4123-90a3-cb50b925359b.png" width=350> <img src="https://user-images.githubusercontent.com/76993392/180642332-c9b5dba7-bec0-423d-867f-c7abb2f663a3.png" width=350>
<br>
<img src="https://user-images.githubusercontent.com/76993392/180642353-dc67b3be-90f6-4419-a5e8-1356a309675c.png" width=350> <img src="https://user-images.githubusercontent.com/76993392/180642357-4cc86807-ea60-4dfb-b7cd-b65ec6e1ab2f.png" width=350>

<hr/>

## リアルタイム判定(Jetson)

|Network|Type|読み込み時間[s]|
|--|:--:|:--:|
|E-Unet+PWConv|float32|25.1587|
|E-Unet+PWConv(FlatBuffers)|int8|0.00161796|
|E-Unet+PWConv(FlatBuffers)|float32|**0.00138240**|

|Network|Size|fps[Hz]|fps*Size|
|--|:--:|:--:|:--:|
|ResNet18|270\*270\*1\*24|1.48|**3,548,448**|
|E-Unet+PWConv|576\*576\*3|**1.83**|1,821,450|
|E-Unet+PWConv(FlatBuffers:f32)|576\*576\*3|1.24|1,234,207|

<hr/>

## プログラムの内容

train = concrete_compaction/KerasFramework-master_tf2/train

- train/SemanticSegmentation.py
  - Semantic Segmentationに関するネットワークを定義
  - 実装してあるもの
    - U-Net
    - PSP-Net    # 未実験
    - [モデル] + Conv + SphereFace/ArcFace/CosFaceの学習モデル
    - [モデル] + Classifier(PWConv + Softmax)の学習モデル
- train/SemSegLight.py
  - 軽量なSegmentationモデルの定義
  - 実装してあるもの
    - E-Unet
    - E-Unet(Softmax) + PWConv
    - E-Unet(Attention, Softmax) + PWConv    # 自作。模索中。
    - ERF-PSPNet
    - CFPNet-M    # 調整中
- train/SemanticSegmentationTrain.py
  - Semantic Segmentationの学習
- train/SemanticSegmentationTest.py
  - Semantic Segmentationのテスト
- train/MyUtil.py
  - モデルの呼び出しや画像の加工などの処理
  - F値や終了時間の予測などの計算、フレッシュ性状データの加工、正解ラベルの定義もここ
- train/Normalization.py
  - 正規化の定義(TensorFlowにないから自作してるもの)
  - 実装してあるもの
    - Batch Renorm
    - Batch Instance Norm
- [TensorFlow Addonsのgithub](https://github.com/tensorflow/addons)から持ってきたもの
  - train/tfa_AdaBelief.py
    - AdaBelief
    - Adamの単純進化
  - train/tfa_RectifiedAdam.py
    - Rectified Adam (RAdam)
    - ステップ数によって変化する適応学習率を取り入れたAdam
- train/lite_model_convertor.py
  - モデル軽量化に関するあれこれ
  - 実装してあるもの
    - モデル量子化


<hr/>


## Docker関連
- Dockerコンテナのビルド
```sh
$ sh docker_build.sh [タグ名]
```
- Dockerコンテナの立ち上げ
```sh
$ sh launch.sh [使用するGPUの番号]
```


<hr/>


## プログラムの実行
### 1. 学習・テストのプラン作成
  train/make_learning_plan.pyを編集（引数の名前で何の設定か分かるようになっているはず）<br>
  <br>
  新しい設定を追加したいときは、train/learning_plan_encoder.pyを編集の上、train/SemanticSegmentationTrain.py、train/SemanticSegmentationTest.pyに新たな引数の受け取りを追加する
### 2. 学習・テストの実行
  Dockerコンテナ内で下記コマンドを打つ
  - 1.で作った学習プランを実行可能にする（以下、エンコードと呼ぶ）
  ```
  $ make
  ```
  - エンコードした学習プランの実行
  ```
  $ learn
  ```
  **learnコマンドを実行すると重みや学習結果など諸々を初期化するので、必ずSAVE_IDが他と被っていないか要確認。**
  - エンコードした学習プランの確認
  ```
  $ plan
  ```


<hr/>


```python

limit = 100_000_000

# π = 4 ( 1 - 1/3 + 1/5 - 1/7 + 1/9 - 1/11 + 1/13 + ... )
pi = sum([4. / ((2*i)+1) * (-2*(i%2)+1) for i in range(limit)])

print(f"{pi = }")
# pi = 3.141592643589326

```
