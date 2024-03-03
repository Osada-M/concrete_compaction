# FROM tensorflow/tensorflow:2.4.0-gpu
FROM nvcr.io/nvidia/tensorflow:21.09-tf2-py3

RUN apt-get update && apt-get upgrade -y && apt-get -y install gosu vim graphviz v4l-utils ffmpeg
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

USER root

RUN apt-get install -y libgl1-mesa-dev

RUN pip install --upgrade pip

# RUN pip uninstall -y tensorflow

RUN pip install numpy==1.19.2
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install colorama
RUN pip install pillow
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install pyyaml
RUN pip install argparse
RUN pip install pydot
RUN pip install tqdm
RUN pip install line-bot-sdk
RUN pip install sklearn
RUN pip install umap-learn
RUN pip install scikit-image
RUN pip install openpyxl
RUN pip install tensorflow_model_optimization
RUN pip install tensorflow==2.10.0

# COPY entrypoint.sh /root/entrypoint.sh
# RUN chmod +x /root/entrypoint.sh
# ENTRYPOINT ["/root/entrypoint.sh"]
