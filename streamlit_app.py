import streamlit as st
import numpy as np
import keras.models
from WeatherSensorAPI import WeatherSensorAPI

# Загрузка нейронных моделей
temp_model = keras.models.load_model("temp_full.keras")
weather_model = keras.models.load_model("weather_full.keras")

# Кодирование типов погоды
weather_encoding = {"морось": 0, "дождь": 1, "солнце": 2, "снег": 3, "туман": 4}
reverse_weather_encoding = {v: k for k, v in weather_encoding.items()}

# Константы
HISTORY = 4  # Количество дней для прогнозирования

# Создание экземпляра датчика
sensor = WeatherSensorAPI()

def main():
    st.title('🌤️ Прогноз погоды')

    # Получаем данные от датчика
    forecast = sensor.get_forecast_for_4_days()

    st.subheader('Прогноз от датчика')
    for i, day_data in enumerate(forecast):
        st.write(f"День {i + 1}: {day_data['temperature']}°C, {day_data['weather_type']}")

    # Получение данных температуры для модели
    temp_inputs = np.array([day['temperature'] for day in forecast]).reshape(1, -1)

    # Прогноз температуры на следующий день
    temp_prediction = temp_model.predict(temp_inputs)
    predicted_temp = int(temp_prediction[0][0])

    # Получение кодированных типов погоды для модели
    coded_weather = np.array([weather_encoding[day['weather_type']] for day in forecast]).reshape(1, -1)

    # Прогноз типа погоды на следующий день
    weather_prediction = weather_model.predict(coded_weather)
    predicted_weather_index = np.argmax(weather_prediction)
    predicted_weather = reverse_weather_encoding[predicted_weather_index]

    # Вывод результатов
    st.subheader('Прогноз на следующий день')
    st.write(f'🌡️ Температура: {predicted_temp}°C')
    st.write(f'🌤️ Тип погоды: {predicted_weather}')

if __name__ == "__main__":
    main()
