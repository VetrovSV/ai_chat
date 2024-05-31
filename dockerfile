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
run useradd -ms /bin/bash user
run echo "user:user" | chpasswd

cmd ["/usr/sbin/sshd", "-D"]

# docker build -t <имя_образа> <путь к каталогу c dockerfile>
