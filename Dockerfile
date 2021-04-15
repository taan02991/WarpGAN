FROM python:3.7

WORKDIR /app

COPY . /app

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

CMD ["python", "main.py"]