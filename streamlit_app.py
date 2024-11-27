import streamlit as st
import numpy as np
import keras.models
from WeatherSensorAPI import WeatherSensorAPI

# Загрузка нейронных моделей
temp_model = keras.models.load_model("temp_full.keras")
weather_model = keras.models.load_model("weather_full.keras")

# Кодирование типов погоды
weather_encoding = {"морось": 0, "дождь": 1, "солнце": 2, "снег": 3, "туман": 4}
weather_encoding_ru_text = ["морось", "дождь", "солнце", "снег", "туман"]
reverse_weather_encoding = {v: k for k, v in weather_encoding.items()}

# Константы
HISTORY = 4  # Количество дней для прогнозирования

# Создание экземпляра датчика
sensor = WeatherSensorAPI()


def main():
    st.title('🌤️ Прогноз погоды')

    # Инициализация переменных для хранения значений формы
    temp_inputs = [st.session_state.get(f"temp_input_{i}", f"{15 + i * 2}") for i in range(HISTORY)]
    weather_inputs = [st.session_state.get(f"weather_input_{i}", "солнце") for i in range(HISTORY)]

    # Основная форма
    with st.form(key="main_form"):
        st.subheader('Какая погода была в предыдущие дни? 📅')
        st.caption('Введите температуру предыдущих дней 🌡️:')

        # Поля для ввода температуры
        temp_cols = st.columns(HISTORY)
        for i, col in enumerate(temp_cols):
            temp_inputs[i] = col.text_input(f'День #{i+1}', value=temp_inputs[i], key=f"temp_input_{i}")

        st.caption('Введите тип погоды 🌤️:')
        # Поля для ввода типа погоды
        weather_cols = st.columns(HISTORY)
        for i, col in enumerate(weather_cols):
            weather_inputs[i] = col.selectbox(f'День #{i+1}', weather_encoding_ru_text, index=weather_encoding_ru_text.index(weather_inputs[i]), key=f"weather_input_{i}")

        # Кнопка для выполнения предсказания
        submit_button = st.form_submit_button(label="Предсказать ✨")

    # Кнопка для подстановки данных из датчика
    if st.button("Получить данные с датчика 📡"):
        forecast = sensor.get_forecast_for_4_days()
        for i in range(HISTORY):
            st.session_state[f"temp_input_{i}"] = str(forecast[i]["temperature"])
            st.session_state[f"weather_input_{i}"] = forecast[i]["weather_type"]

        st.experimental_rerun()  # Перезагрузка страницы для обновления значений

    # Обработка предсказания после нажатия кнопки
    if submit_button:
        # Преобразование входных данных для модели
        temp_inputs_float = [float(x) for x in temp_inputs]
        temp_inputs_array = np.expand_dims(temp_inputs_float, axis=0)
        temp_result = temp_model.predict(temp_inputs_array)

        coded_weather = [weather_encoding[w] for w in weather_inputs]
        coded_weather_array = np.expand_dims(coded_weather, axis=0)
        predicted_weather_index = np.argmax(weather_model.predict(coded_weather_array))
        predicted_weather = reverse_weather_encoding[predicted_weather_index]

        # Вывод результатов
        st.subheader('Прогноз на следующий день 📅')
        st.write(f'🌡️ Температура: **{int(temp_result[0][0])}°C**')
        st.write(f'🌤️ Тип погоды: **{predicted_weather}**')


if __name__ == "__main__":
    main()
