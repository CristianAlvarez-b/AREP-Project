import requests

def get_weather_forecast(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url).json()

    return {
        'temp': response['main']['temp'],
        'lluvia': response.get('rain', {}).get('1h', 0.0) or 0.0
    }
