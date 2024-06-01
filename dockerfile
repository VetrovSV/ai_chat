# from debian:bookworm-slim
from python:3.12.3-bullseye
# нужен VPN

run apt update && apt install -y \
  openssh-server \
  bash \
  && rm -rf /var/lib/apt/lists/*


RUN adduser --disabled-password --home /home/app --shell /bin/bash app

USER app
WORKDIR /home/app

COPY requirements.txt requirements.txt


RUN pip3 install --no-cache-dir -r requirements.txt
# может занимать 1200+ секунд

COPY ["ygpt_secret.txt", "."]
run mkdir data
COPY ["data/", "data/"]
COPY ["*.py", "."]
# перенос модели текстовых эмбеддингов
run mkdir models--cointegrated--LaBSE-en-ru
COPY ["models--cointegrated--LaBSE-en-ru/", "models--cointegrated--LaBSE-en-ru"]

# ENV HOST=0.0.0.0
# ENV PORT=60004

# проверка сервера: curl -X POST -H 'Content-Type: application/json' -d '{"query":"Hello"}'  http://0.0.0.0:8000/assist

cmd python3 main.py --port 60004 --host 0.0.0.0


# todo: скачать модель для эмбеддингов заранее
# todo: задать версии, привести в порядок папки и т.п.
# todo: оптимизировать размер образа

# docker build -t <имя_образа> <путь к каталогу c dockerfile>
# docker build -t chat_ai:test_server6 .
# docker run  -p 60004:60004 chat_ai:test_server4