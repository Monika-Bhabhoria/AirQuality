FROM python:3.12-rc-slim-buster
WORKDIR /app
COPY . /

RUN apt update -y && apt install awscli -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y && pip install -r requirements.txt
CMD ["python", "app.py"]