## Структура


- `main.py` - главный файл сервера, тут обработчики запросов по API
- `main_cli_chat.py` - главный файл, можно потестить чат в консоли 
- `chat_bot.py` - туда спрятал все функции загрузки, инициализации и т.п.
- `data` - папка с данными
  - `dataset.json` - сырой датасет 
  - `dataset.faiss` - файлы векторной БД
- `fastAPI` - примеры API
- `experiments\checking_bot.py` - файл с проверкой точности ответов бота
- `openapi.yaml`,`openapi.yaml` - контракт swagger'а, определяющего правило взаимодействия нашего развернутого решения по API для проверки
- `models.py` - файл определяющий модели данных для API с использованием `pydantic`.

 
[Скринкаст](https://drive.google.com/file/d/1psd_ouyXrp1EoJtI7dBaFsAFRzRUMesj/view?usp=drive_link)

## Развёртывание

Готовый прототип: [Docker-образ](https://drive.google.com/file/d/145VOKdAfwvT-sv0KIqvn-3sMFiVwFCEA/view?usp=drive_link)

**Создание образа**
0. Клонировать репозиторий
1. Настроить переменные для доступа к YandexGPT
2. Собрать образ
```bash
docker build -t chat_ai:test_server8 .
```
3. Запустить контейнер
```bash
docker run --restart=always  -p 1234:60004 chat_ai:test_server8  
```

Для полноценной работы сервера нужно запустить сервер ollama, на который загружена LLM, указанная в  `chat_bot.py`, 
в переменной `LLM_NAME`


## Данные для  доустпа

Адрес сервера: http://95.189.96.144:60012/assist

Пример запроса:  
`curl -X POST -H 'Content-Type: application/json' -d '{"query":"Как получить статистику пополнения счёта?"}'  http://0.0.0.0:60011/assist`

*количество запросов в час может быть ограничено (до 100) используемым сервисом YandexGPT*

