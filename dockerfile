# from debian:bookworm-slim
from python:3.12.3-bullseye
run apt update && apt install -y \
  openssh-server \
  bash \
  && rm -rf /var/lib/apt/lists/*
# todo: python, jupyter

# включить авторизацию по паролю
# папка для работы sshd
run echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
run mkdir /var/run/sshd
run ssh-keygen -A

# задать пароль пользователя
# run useradd -ms /bin/bash user
# RUN adduser --home /home/app --shell /bin/bash app
# run echo "app:app" | chpasswd

RUN adduser --disabled-password --home /home/app --shell /bin/bash app

USER app
WORKDIR /home/app

COPY requirements_server_test.txt requirements.txt
RUN pip3 install -r requirements.txt
# RUN python -m venv app_venv
# похоже это костыль, создавать виртуальное окружение, но так быстрее настроить
# RUN source app_venv/bin/acivate

COPY ["testAPI/Client_Server (Gena)/*", "."]
# ADD templates/* /home/app/templates/


# проверка сервера: curl -X POST -H 'Content-Type: application/json' -d '{"query":"Hello"}'  http://0.0.0.0:8000/assist


# cmd ["/usr/sbin/sshd", "-D"]
cmd python3 main.py


# docker build -t <имя_образа> <путь к каталогу c dockerfile>
# docker run  -p 8000:8000