import streamlit as st
import numpy as np
import sqlite3
import keras.models
from WeatherSensorAPI import WeatherSensorAPI

# Загрузка нейронных моделей
temp_model = keras.models.load_model("temp_full.keras")
weather_model = keras.models.load_model("weather_full.keras")

# Кодирование типов погоды
weather_encoding = {"морось": 0, "дождь": 1, "солнце": 2, "снег": 3, "туман": 4}
reverse_weather_encoding = {v: k for k, v in weather_encoding.items()}
weather_encoding_ru_text = ["морось", "дождь", "солнце", "снег", "туман"]

# Константы
HISTORY = 4  # Количество дней для прогнозирования

# Создание экземпляра датчика
sensor = WeatherSensorAPI()

# Инициализация session_state
if "temp_inputs" not in st.session_state:
    st.session_state["temp_inputs"] = [f"{15 + i * 2}" for i in range(HISTORY)]
if "weather_inputs" not in st.session_state:
    st.session_state["weather_inputs"] = ["солнце"] * HISTORY

if "rerun" in st.session_state and st.session_state["rerun"]:
    st.session_state["rerun"] = False
    st.experimental_rerun()


# Функция для работы с базой данных
def init_db():
    conn = sqlite3.connect("weather_predictions.db")
    cursor = conn.cursor()

    # Пересоздание таблицы
    cursor.execute("DROP TABLE IF EXISTS predictions")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temp_inputs TEXT,
            weather_inputs TEXT,
            temperature_prediction TEXT,
            weather_prediction TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_prediction(temp_inputs, weather_inputs, temp_prediction, weather_prediction):
    conn = sqlite3.connect("weather_predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (temp_inputs, weather_inputs, temperature_prediction, weather_prediction) VALUES (?, ?, ?, ?)",
        (temp_inputs, weather_inputs, temp_prediction, weather_prediction)
    )
    conn.commit()
    conn.close()


def get_predictions():
    conn = sqlite3.connect("weather_predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, temp_inputs, weather_inputs, temperature_prediction, weather_prediction FROM predictions ORDER BY id DESC"
    )
    predictions = cursor.fetchall()
    conn.close()
    return predictions


def delete_prediction(prediction_id):
    conn = sqlite3.connect("weather_predictions.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    conn.commit()
    conn.close()


def update_inputs_from_sensor():
    """Обновить значения формы данными из сенсора."""
    forecast = sensor.get_forecast_for_4_days()
    for i in range(HISTORY):
        st.session_state["temp_inputs"][i] = str(forecast[i]["temperature"])
        st.session_state["weather_inputs"][i] = forecast[i]["weather_type"]


def main():
    # Инициализация базы данных
    init_db()

    st.title('🌤️ Прогноз погоды')

    # Кнопка для получения данных с датчика перед формой
    st.button("Получить данные с датчика 📡", on_click=update_inputs_from_sensor)

    # Основная форма
    with st.form(key="main_form"):
        st.subheader('Какая погода была в предыдущие дни? 📅')
        st.caption('Введите температуру предыдущих дней 🌡️:')

        # Поля для ввода температуры
        temp_cols = st.columns(HISTORY)
        for i, col in enumerate(temp_cols):
            st.session_state["temp_inputs"][i] = col.text_input(
                f'День #{i + 1}',
                value=st.session_state["temp_inputs"][i],
                key=f"temp_input_{i}"
            )

        st.caption('Введите тип погоды 🌤️:')
        # Поля для ввода типа погоды
        weather_cols = st.columns(HISTORY)
        for i, col in enumerate(weather_cols):
            st.session_state["weather_inputs"][i] = col.selectbox(
                f'День #{i + 1}',
                weather_encoding_ru_text,
                index=weather_encoding_ru_text.index(st.session_state["weather_inputs"][i]),
                key=f"weather_input_{i}"
            )

        # Кнопка для выполнения предсказания
        submit_button = st.form_submit_button(label="Предсказать ✨")

    # Обработка предсказания после нажатия кнопки
    if submit_button:
        # Преобразование входных данных для модели
        temp_inputs_float = [float(x) for x in st.session_state["temp_inputs"]]
        temp_inputs_array = np.expand_dims(temp_inputs_float, axis=0)
        temp_result = temp_model.predict(temp_inputs_array)

        coded_weather = [weather_encoding[w] for w in st.session_state["weather_inputs"]]
        coded_weather_array = np.expand_dims(coded_weather, axis=0)
        predicted_weather_index = np.argmax(weather_model.predict(coded_weather_array))
        predicted_weather = reverse_weather_encoding[predicted_weather_index]

        # Сохранение результата в базу данных
        insert_prediction(
            str(st.session_state["temp_inputs"]),
            str(st.session_state["weather_inputs"]),
            f"{int(temp_result[0][0])}°C",
            predicted_weather
        )

        # Вывод результатов
        st.subheader('Прогноз на следующий день 📅')
        st.write(f'🌡️ Температура: **{int(temp_result[0][0])}°C**')
        st.write(f'🌤️ Тип погоды: **{predicted_weather}**')

    # Отображение последних прогнозов
    st.subheader("История прогнозов 📝")
    predictions = get_predictions()
    if predictions:
        for prediction_id, temp_inputs, weather_inputs, temp_prediction, weather_prediction in predictions:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**Изначальные данные:** 🌡️ {temp_inputs}, 🌤️ {weather_inputs}")
                st.write(f"**Прогноз:** 🌡️ {temp_prediction}, 🌤️ {weather_prediction}")
            with col2:
                if st.button("❌ Удалить", key=f"delete_{prediction_id}"):
                    delete_prediction(prediction_id)
                    st.session_state["rerun"] = True


if __name__ == "__main__":
    main()
