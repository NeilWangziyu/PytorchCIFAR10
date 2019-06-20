FROM ufoym/deepo

MAINTAINER t-ziw@microsoft.com

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

COPY * /app/

#ENTRYPOINT /bin/bash

ENV ENVIRONMENT local

ENTRYPOINT python ./main.py --GPU False --model resnet


