#!/bin/bash

USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

echo "Starting with UID : $USER_ID, GID: $GROUP_ID"
useradd -u $USER_ID -o -m user
groupmod -o -g $GROUP_ID user
export HOME=/home/user


echo "alias make='python /workspace/train/make_learning_plan.py'" >> /home/user/.bashrc
echo "alias learn='sh /workspace/train/learning_plan.sh'" >> /home/user/.bashrc
echo "alias plan='more /workspace/train/learning_plan.sh'" >> /home/user/.bashrc
echo "export TF_CPP_MIN_LOG_LEVEL=2" >> /home/user/.bashrc

. /home/user/.bashrc

exec /usr/sbin/gosu user "$@"
