import requests

# URL вашего API
api_url = "http://localhost:60004/assist"

# Данные запроса
request_data = {
    "query": "Какого вопроса нет в базе знаний тинькофф?"
}

# Отправка POST-запроса
response = requests.post(api_url, json=request_data)

# Проверка успешности запроса
if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Failed to get response:", response.status_code)
    print("Response:", response.json())


# ещё проверить сервер можно так:
# curl -X POST -H 'Content-Type: application/json' -d '{"query":"Hello"}'  http://0.0.0.0:8000/assist