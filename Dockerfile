FROM bvlc/caffe:gpu
LABEL maintainer AusCoder

RUN python -c "import caffe"

COPY ./requirements.txt /root/app/requirements.txt
WORKDIR /root/app
RUN pip install -r requirements.txt

COPY . /root/app

CMD python -c "print('sup dog')"