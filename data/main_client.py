#import requests
#import json


#def get_all_post():
    #response = requests.get('https://api-inference.huggingface.co/models/gpt2').json()
    #print(response)

#def post_new_post():
    #response= requests.post()

#get_all_post()


# import requests
#
# def get_model_prediction(question):
#     # Адрес API модели на другом компьютере
#     model_api_url = 'https://api-inference.huggingface.co/models/gpt2'
#
#     # Параметры запроса
#     payload = {'question': question}

    # try:
    #     # Отправляем GET запрос к модели
    #     response = requests.get(model_api_url, params=payload)
    #     # Проверяем статус код ответа
    #     if response.status_code == 200:
    #         # Возвращаем ответ от модели
    #         return response.json()['prediction']
    #     else:
    #         return "Ошибка при получении ответа от модели. Статус код: {}".format(response.status_code)
    # except Exception as e:
    #     return "Ошибка при отправке запроса к модели: {}".format(e)

# Пример использования
# question = "Какая будет погода завтра?"
# prediction = get_model_prediction(question)
# print("Ответ модели:", prediction)



#import requests

#def get_model_prediction(question):
    # Адрес API модели на другом компьютере
    #model_api_url = 'https://api-inference.huggingface.co/models/gpt2'

    # Параметры запроса
   # payload = {'question': question}

   ## try:
        # Отправляем GET запрос к модели
       # response = requests.get(model_api_url, params=payload)
        # Проверяем статус код ответа
        #if response.status_code == 200:
            # Возвращаем ответ от модели
         #   return response.json()['prediction']
        #else:
          #  return "Ошибка при получении ответа от модели. Статус код: {}".format(response.status_code)
  #  except Exception as e:
 #       return "Ошибка при отправке запроса к модели: {}".format(e)

# Пример использования
#question = "Какая будет погода завтра?"
#prediction = get_model_prediction(question)
#print("Ответ модели:", prediction)


import requests
import json

# URL API
API_URL = "http://127.0.0.1:8000/assist"


def test_api():
    # Тестовый запрос
    test_query = "hello"

    # Формирование данных запроса
    request_data = {
        "query": test_query
    }

    
    response = requests.post(API_URL, json=request_data)

    if response.status_code == 200:
        response_data = response.json()
        print("API:")
        print(f"respons: {response_data['text']}")
        print(f"si: {', '.join(response_data['links'])}")
    else:
        print(f"error: {response.status_code}")
        if response.status_code == 422:
            validation_error = response.json()
            print(f"error: {validation_error['detail']}")


# Вызов функции для тестирования API
test_api()





