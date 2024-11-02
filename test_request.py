import requests

url = "http://localhost:8000/predict"
data = {
    "system": "You are a helpful assistant, which gives concise and short answers",
    "user": "Why can we say that pydamids were not built by aliens?",
    'generation_config': {
        'temperature': 0.0,
        "max_new_tokens": 500,
        "return_full_text": False,
        }
    }

response = requests.post(url, json=data)

if response.status_code == 200:
    response = response.json()
    print(response[0]['generated_text'])
else:
    print("Request failed:", response.status_code, response.text)
