import requests

url = "http://localhost:8000/predict"
data = {
    "system": """You are a helpful mental health assistant, who cares about providing necessary advices for 
    people with compilated life situations
    """,
    "user": "I am afraid of not being good enough at new work",
    'generation_config': {
        'temperature': 0.0,
        "max_new_tokens": 1000,
        "return_full_text": False,
        }
    }

response = requests.post(url, json=data)

if response.status_code == 200:
    response = response.json()
    print(response[0]['generated_text'])
else:
    print("Request failed:", response.status_code, response.text)
