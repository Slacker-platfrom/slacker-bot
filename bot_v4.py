from typing import Dict

import os
import logging
import telebot
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

if not TELEGRAM_TOKEN or not OPENAI_API_KEY or not DEEPSEEK_API_KEY:
    raise ValueError("Необходимо задать TELEGRAM_TOKEN, OPENAI_API_KEY и DEEPSEEK_API_KEY в файле .env")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

bot = telebot.TeleBot(TELEGRAM_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)

# Хранит выбранные модели для каждого пользователя (DeepSeek, gpt-4o, gpt-4o-mini, gpt-3.5-turbo и т.д.)
user_models: Dict[int, str] = {}
# Временные флаги для отслеживания состояния выбора
temp_flags: Dict[int, str] = {}


def query_deepseek(user_text: str) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": user_text}]
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def query_chatgpt(user_text: str, model_name: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_text}],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()


@bot.message_handler(commands=['start'])
def start_message(message: telebot.types.Message):
    welcome_text = (
        "Привет! Я бот, который может общаться с DeepSeek или ChatGPT.\n"
        "Задавай свои вопросы или используй команду /help для списка доступных команд."
    )
    bot.send_message(message.chat.id, welcome_text)


@bot.message_handler(commands=['help'])
def help_message(message: telebot.types.Message):
    help_text = (
        "Доступные команды:\n"
        "/start - запуск бота\n"
        "/help - помощь\n"
        "/choose_model - выбор модели (DeepSeek или ChatGPT)\n"
        "Для ChatGPT доступны модели: gpt-4, gpt-3.5-turbo\n"
        "Просто отправь сообщение, и я отвечу, используя выбранную модель."
    )
    bot.send_message(message.chat.id, help_text)


@bot.message_handler(commands=['choose_model'])
def choose_model(message: telebot.types.Message):
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True)
    markup.add('DeepSeek', 'ChatGPT')
    bot.send_message(message.chat.id, "Сначала выбери провайдера:", reply_markup=markup)
    temp_flags[message.from_user.id] = 'choosing_provider'


@bot.message_handler(func=lambda msg: msg.text in ['DeepSeek', 'ChatGPT'])
def handle_provider_choice(message: telebot.types.Message):
    user_id = message.from_user.id
    if temp_flags.get(user_id) == 'choosing_provider':
        if message.text == 'ChatGPT':
            markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True)
            markup.add('gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4o')
            bot.send_message(message.chat.id, "Теперь выбери модель ChatGPT:", reply_markup=markup)
            temp_flags[user_id] = 'choosing_chatgpt_model'
        else:
            user_models[user_id] = 'DeepSeek'
            bot.send_message(message.chat.id, "Выбрана модель DeepSeek!")
            del temp_flags[user_id]


@bot.message_handler(func=lambda msg: msg.text in ['gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4o'])
def handle_model_choice(message: telebot.types.Message):
    user_id = message.from_user.id
    if temp_flags.get(user_id) == 'choosing_chatgpt_model':
        user_models[user_id] = message.text
        bot.send_message(message.chat.id, f"Выбрана модель {message.text}!")
        del temp_flags[user_id]


@bot.message_handler(func=lambda msg: True)
def reply_message(message: telebot.types.Message):
    try:
        user_id = message.from_user.id
        user_text = message.text
        
        # Если пользователь в процессе выбора модели
        if temp_flags.get(user_id):
            bot.send_message(message.chat.id, "Пожалуйста, заверши выбор модели через меню.")
            return

        model_choice = user_models.get(user_id, 'gpt-4o')  # По умолчанию gpt-4o

        if model_choice == 'DeepSeek':
            reply = query_deepseek(user_text)
        else:
            reply = query_chatgpt(user_text, model_choice)

        bot.send_message(message.chat.id, reply)
    except Exception as e:
        logging.error("Ошибка при обработке запроса: %s", e, exc_info=True)
        bot.send_message(message.chat.id, "Произошла ошибка. Попробуйте позже.")


if __name__ == "__main__":
    logging.info("Бот запущен...")
    bot.polling(none_stop=True)
