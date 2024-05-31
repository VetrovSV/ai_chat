import requests

# URL вашего API
api_url = "http://127.0.0.1:8000/assist"

# Данные запроса
request_data = {
    "query": "Hello"
}

# Отправка POST-запроса
response = requests.post(api_url, json=request_data)

# Проверка успешности запроса
if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Failed to get response:", response.status_code)
    print("Response:", response.json())
