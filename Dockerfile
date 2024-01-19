FROM python:3.9.18-bookworm

EXPOSE 8000

RUN git clone https://gitlab.ussc.ru/mkoltunov/document_ner_extraction
RUN apt-get update && \
apt-get install gcc && \
apt install g++

RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

WORKDIR ./document_ner_extraction

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

COPY ./layoutxlm-base ./layoutxlm-base
COPY ./layoutxlm-finetuned-doc ./layoutxlm-finetuned-doc

CMD ["python", "main.py"]
