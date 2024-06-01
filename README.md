## Структура

- `main.py` - главный файл, тут обработчики запросов по API
- `main_cli_chat.py` - главный файл, можно потестить чат в консоли 
- `chat_bot.py` - туда спрятал все функции загрузки, инициализации и т.п.
- `data` - папка с данными
  - `dataset.json` - сырой датасет 
  - `dataset.faiss` - файлы векторной БД
- `fastAPI` - примеры API

Для полноценной работы сервера нужно запустить сервер ollama, на который загружена LLM, указанная в  `chat_bot.py`, 
в переменной `LLM_NAME`

