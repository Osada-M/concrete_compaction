#!/bin/bash

workspace="/media/nagalab/SSD1.7TB/nagalab/osada_ws"
dataset="/media/nagalab/SSD1.7TB/nagalab/Dataset"


if [ $# -ne 1 ]; then
    nvidia-smi
    echo;
    echo "GPUの番号も入力して" 1>&2
    exit 1
fi

# export DISPLAY=:0.0

xhost +

# --privileged \
docker container run \
--rm --gpus "device=$1" -it \
--device=/dev/video0:/dev/video0:rw \
-v $workspace/concrete_compaction/:/workspace/osada_ws \
-v $workspace/concrete_compaction/KerasFramework-master_tf2/train:/workspace/train \
-v $workspace/concrete_compaction/text_dataset/ngc_docker:/workspace/mesh_dataset \
-v $workspace/LuminanceData/:/workspace/luminance \
-v $workspace/semanticSegmentation/:/workspace/semanticSegmentation \
-v $workspace/fullframe/:/workspace/fullframe \
-v $workspace/hidden_layer/:/workspace/hidden \
-v $workspace/explain/:/workspace/explain \
-v $dataset/CompactionVideo/:/workspace/video \
-v $workspace/mesh_encoder_result/:/workspace/mesh_encoder_result \
-v $workspace/cpp/:/workspace/cpp \
-v $dataset:/workspace/Dataset \
-v /home/nagalab/:/workspace/home_nagalab \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
-e DISPLAY=$DISPLAY \
-e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) \
osada:20220707 bash
# osada_line:20220626 bash
# osada:20220331 bash
