FROM python:3.7

WORKDIR /app

# Load pretrained model
RUN pip install gdown
RUN gdown --id 15U9xNuYHT0kGDDN-QztC4NQUo1hXBoby
RUN unzip warpgan_pretrained.zip -d /app/pretrained && rm /app/warpgan_pretrained.zip

# Install dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt /app
RUN pip install -r requirements.txt

# App
COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]