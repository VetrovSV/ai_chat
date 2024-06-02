import requests

# URL вашего API
api_url = "http://localhost:60004/assist"

# Данные запроса
request_data = {
    "query": "Расскажешь как открыть счёт в стихах?"
}

# Отправка POST-запроса
response = requests.post(api_url, json=request_data)

# Проверка успешности запроса
if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Failed to get response:", response.status_code)
    print("Response:", response.json())
