import os
import logging
import telebot
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
    raise ValueError("Необходимо задать TELEGRAM_TOKEN и OPENAI_API_KEY в файле .env")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация бота Telegram и клиента OpenAI
bot = telebot.TeleBot(TELEGRAM_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def start_message(message):
    welcome_text = (
        "Привет! Я бот на базе GPT.\n"
        "Задавай свои вопросы или используй команду /help для списка доступных команд."
    )
    bot.send_message(message.chat.id, welcome_text)

# Обработчик команды /help
@bot.message_handler(commands=['help'])
def help_message(message):
    help_text = (
        "Доступные команды:\n"
        "/start - запуск бота\n"
        "/help - помощь\n"
        "Просто отправь сообщение, и я отвечу, используя GPT."
    )
    bot.send_message(message.chat.id, help_text)

# Обработчик всех текстовых сообщений
@bot.message_handler(func=lambda message: True)
def gpt_reply(message):
    try:
        response = client.chat.completions.create(
            model= "gpt-4o-mini", #"o1-mini", #"o3-mini",   # Или "gpt-3.5-turbo"
            messages=[{"role": "user", "content": message.text}],
            max_tokens=1000
        )
        reply = response.choices[0].message.content.strip()
        bot.send_message(message.chat.id, reply)
    except Exception as e:
        logging.error("Ошибка при обработке запроса: %s", e, exc_info=True)
        bot.send_message(message.chat.id, "Произошла ошибка. Попробуйте позже.")

if __name__ == "__main__":
    logging.info("Бот запущен...")
    bot.polling(none_stop=True)
