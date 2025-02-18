FROM python:3.12-rc-slim-buster
# WORKDIR /app
# COPY . /app

# RUN apt update -y

# RUN apt-get update && pip install -r requirements.txt
# CMD ["python3", "app.py"]
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y && pip install -r requirements.txt
CMD ["python3", "app.py"]