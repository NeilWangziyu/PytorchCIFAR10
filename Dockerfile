FROM ufoym/deepo

MAINTAINER t-ziw@microsoft.com

WORKDIR /app/

COPY requirements.txt /app/

RUN pip install numpy pandas scikit_learn

COPY /models/. /app/models/

COPY . /app/

#RUN ls -la /app/*
#ENTRYPOINT /bin/bash

ENV ENVIRONMENT local

ENTRYPOINT python ./main.py --GPU True --model resnet


