import random

class WeatherSensorAPI:
    """
    API для эмуляции работы датчика температуры и типа погоды
    """

    WEATHER_TYPES = ["солнце", "морось", "дождь", "снег", "туман"]

    def __init__(self, start_temp=10, fluctuation=5):
        """
        Инициализация API датчика.

        :param start_temp: Базовая температура для первого дня (°C)
        :param fluctuation: Максимальная амплитуда колебаний температуры (°C)
        """
        self.current_temp = start_temp
        self.fluctuation = fluctuation

    def get_temperature_for_day(self):
        """
        Возвращает случайную температуру для одного дня

        :return: Целое значение температуры (°C)
        """
        change = random.randint(-self.fluctuation, self.fluctuation)
        self.current_temp += change
        return self.current_temp

    def get_weather_type_for_day(self):
        """
        Возвращает случайный тип погоды для одного дня

        :return: Строка с описанием типа погоды
        """
        return random.choice(self.WEATHER_TYPES)

    def get_forecast_for_4_days(self):
        """
        Возвращает прогноз погоды за 4 дня (температура и тип погоды).

        :return: Список словарей с температурой и типом погоды за каждый день.
        """
        forecast = []
        for _ in range(4):
            forecast.append({
                "temperature": self.get_temperature_for_day(),
                "weather_type": self.get_weather_type_for_day()
            })
        return forecast
