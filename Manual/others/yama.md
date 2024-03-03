# Manual for ROS
### 完全自動締固めシステムの実行ガイド

<hr/>

<br>
<br>

### 完全自動締固めシステムとは
        コンクリートの締固め工程の完全自動化を目指したシステム。
        締固め判定の自動化を行う自動締固めシステムをROS(Robot Operating System)をを用いることで拡張し、カメラ映像の取得やバイブレータの移動なども自動化した。
<img src="ros_system.png" width=100%>

<br>

### システムの実行方法
    システムに必要なコードは6つ
    →frame_read_node.py, predict_node.py, drawing_node.py, convert_node.py, x_move_node.py, y_move_node.py

    以上のコードをlaunch_realtime_judgment_using_tftrt.launchを用いて同時起動させることでシステムが起動する。
    concrete_realtime_judgment_system.desktopを用いることで、デスクトップにショートカットアプリを作成することができる。　

<br>

### コードの保存先
    \\202.13.169.8\disk1\100_研究データ\105_コンクリート\山下\code
