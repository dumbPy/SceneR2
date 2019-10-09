FROM pytorch/pytorch

RUN apt-get update && apt-get install wget && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/dumbPy/SceneR2.git 
WORKDIR SceneR2
RUN pip install -r requirements.txt
RUN cd SceneR2/yolov3/weights && chmod +x ./download_weights.sh && ./download_weights.sh
WORKDIR SceneR2WebApp

CMD python manage.py runserver 0.0.0.0:8000
