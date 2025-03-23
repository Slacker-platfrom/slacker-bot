from typing import Dict, Optional
import os
import logging
import telebot
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
from functools import lru_cache
from anthropic import Anthropic

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if not all([TELEGRAM_TOKEN, OPENAI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY]):
    missing_keys = []
    if not TELEGRAM_TOKEN: missing_keys.append("TELEGRAM_TOKEN")
    if not OPENAI_API_KEY: missing_keys.append("OPENAI_API_KEY")
    if not DEEPSEEK_API_KEY: missing_keys.append("DEEPSEEK_API_KEY")
    if not ANTHROPIC_API_KEY: missing_keys.append("ANTHROPIC_API_KEY")
    raise ValueError(f"Отсутствуют необходимые переменные окружения: {', '.join(missing_keys)}")

# Инициализация клиентов
bot = telebot.TeleBot(TELEGRAM_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Константы
DEFAULT_MODEL = 'deepseek-reasoner'
DEFAULT_PROVIDER = 'DeepSeek'  # Добавляем провайдера по умолчанию
AVAILABLE_CHATGPT_MODELS = ['gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4o']
AVAILABLE_CLAUDE_MODELS = ['claude-3-5-sonnet-20240620', 'claude-3-7-sonnet-20250219', 'claude-3-haiku-20240307']
AVAILABLE_DEEPSEEK_MODELS = ['deepseek-chat', 'deepseek-coder', 'deepseek-reasoner']
MAX_TOKENS = 1000

# Состояния пользователей
class UserState:
    NORMAL = 'normal'
    CHOOSING_PROVIDER = 'choosing_provider'
    CHOOSING_MODEL = 'choosing_chatgpt_model'
    CHOOSING_CLAUDE_MODEL = 'choosing_claude_model'
    CHOOSING_DEEPSEEK_MODEL = 'choosing_deepseek_model'

# Хранение данных пользователей
class UserData:
    def __init__(self):
        self.models: Dict[int, str] = {}  # id пользователя -> модель
        self.states: Dict[int, str] = {}  # id пользователя -> состояние
        self.providers: Dict[int, str] = {}  # id пользователя -> провайдер (OpenAI, DeepSeek, Anthropic)
    
    def get_model(self, user_id: int) -> str:
        """Получить текущую модель пользователя или вернуть модель по умолчанию"""
        return self.models.get(user_id, DEFAULT_MODEL)
    
    def set_model(self, user_id: int, model: str) -> None:
        """Установить модель для пользователя"""
        self.models[user_id] = model
    
    def get_state(self, user_id: int) -> str:
        """Получить текущее состояние пользователя"""
        return self.states.get(user_id, UserState.NORMAL)
    
    def set_state(self, user_id: int, state: str) -> None:
        """Установить состояние для пользователя"""
        self.states[user_id] = state
    
    def reset_state(self, user_id: int) -> None:
        """Сбросить состояние пользователя в нормальное"""
        if user_id in self.states:
            self.states[user_id] = UserState.NORMAL
            
    def get_provider(self, user_id: int) -> str:
        """Получить текущего провайдера пользователя или вернуть провайдера по умолчанию"""
        return self.providers.get(user_id, DEFAULT_PROVIDER)
    
    def set_provider(self, user_id: int, provider: str) -> None:
        """Установить провайдера для пользователя"""
        self.providers[user_id] = provider

user_data = UserData()

# API запросы с обработкой ошибок
def query_deepseek(user_text: str, model_name: str) -> str:
    """Отправить запрос к DeepSeek API с обработкой ошибок"""
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": user_text}]
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        logger.error(f"Ошибка запроса к DeepSeek API: {e}")
        return f"Ошибка при обращении к DeepSeek: {str(e)}"
    except (KeyError, IndexError) as e:
        logger.error(f"Ошибка парсинга ответа DeepSeek: {e}")
        return "Ошибка в формате ответа от DeepSeek"

def query_chatgpt(user_text: str, model_name: str) -> str:
    """Отправить запрос к OpenAI API с обработкой ошибок"""
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": user_text}],
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка запроса к OpenAI API ({model_name}): {e}")
        return f"Ошибка при обращении к модели {model_name}: {str(e)}"

def query_claude(user_text: str, model_name: str) -> str:
    """Отправить запрос к Claude API с обработкой ошибок"""
    try:
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": user_text}
            ]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Ошибка запроса к Anthropic API ({model_name}): {e}")
        return f"Ошибка при обращении к модели {model_name}: {str(e)}"

# Создание инлайн-клавиатур
def create_provider_keyboard():
    """Создать инлайн-клавиатуру для выбора провайдера"""
    markup = telebot.types.InlineKeyboardMarkup(row_width=1)
    deepseek_btn = telebot.types.InlineKeyboardButton("DeepSeek", callback_data="provider_DeepSeek")
    chatgpt_btn = telebot.types.InlineKeyboardButton("ChatGPT", callback_data="provider_ChatGPT")
    claude_btn = telebot.types.InlineKeyboardButton("Claude", callback_data="provider_Claude")
    markup.add(deepseek_btn, chatgpt_btn, claude_btn)
    return markup

def create_model_keyboard(provider):
    """Создать инлайн-клавиатуру для выбора модели в зависимости от провайдера"""
    markup = telebot.types.InlineKeyboardMarkup(row_width=1)
    
    if provider == "DeepSeek":
        models = AVAILABLE_DEEPSEEK_MODELS
        prefix = "deepseek_model_"
    elif provider == "ChatGPT":
        models = AVAILABLE_CHATGPT_MODELS
        prefix = "chatgpt_model_"
    elif provider == "Claude":
        models = AVAILABLE_CLAUDE_MODELS
        prefix = "claude_model_"
    else:
        return markup  # Пустая клавиатура в случае ошибки
    
    buttons = [telebot.types.InlineKeyboardButton(model, callback_data=f"{prefix}{model}") for model in models]
    for button in buttons:
        markup.add(button)
    
    return markup

# Команда для удаления клавиатуры
@bot.message_handler(commands=['clear_keyboard'])
def clear_keyboard(message: telebot.types.Message):
    """Удаляет клавиатуру"""
    bot.send_message(message.chat.id, "Клавиатура удалена", 
                    reply_markup=telebot.types.ReplyKeyboardRemove())

# Кнопка быстрого выбора модели
@bot.message_handler(commands=['quick_model'])
def quick_model_selection(message: telebot.types.Message):
    """Быстрый выбор модели через инлайн-кнопки"""
    markup = telebot.types.InlineKeyboardMarkup(row_width=1)
    
    # Кнопка для выбора провайдера
    providers_btn = telebot.types.InlineKeyboardButton(
        "📋 Выбрать провайдера", 
        callback_data="select_provider"
    )
    
    # Кнопки для быстрого выбора популярных моделей
    gpt4o_btn = telebot.types.InlineKeyboardButton(
        "💬 GPT-4o (OpenAI)", 
        callback_data="quick_model_ChatGPT_gpt-4o"
    )
    claude_btn = telebot.types.InlineKeyboardButton(
        "🧠 Claude 3.7 Sonnet", 
        callback_data="quick_model_Claude_claude-3-7-sonnet-20250219"
    )
    deepseek_reasoner_btn = telebot.types.InlineKeyboardButton(
        "🔍 DeepSeek Reasoner", 
        callback_data="quick_model_DeepSeek_deepseek-reasoner"
    )
    
    markup.add(providers_btn, gpt4o_btn, claude_btn, deepseek_reasoner_btn)
    
    bot.send_message(
        message.chat.id, 
        "🤖 *Выбор модели AI*\n\nВыберите провайдера или одну из популярных моделей:", 
        parse_mode="Markdown",
        reply_markup=markup
    )

# Обработчики команд
@bot.message_handler(commands=['start'])
def start_message(message: telebot.types.Message):
    """Обработчик команды /start"""
    welcome_text = (
        "Привет! Я бот, который может общаться с DeepSeek, ChatGPT или Claude.\n"
        "По умолчанию используется модель DeepSeek Reasoner.\n\n"
        "Задавай свои вопросы или используй команду /help для списка доступных команд.\n\n"
        "📌 Новая команда: /quick_model - быстрый выбор модели через кнопки\n"
        "🗑️ Проблемы с клавиатурой? Используйте /clear_keyboard для её удаления"
    )
    # Явное удаление клавиатуры при начале работы с ботом
    bot.send_message(message.chat.id, welcome_text, reply_markup=telebot.types.ReplyKeyboardRemove())

@bot.message_handler(commands=['help'])
def help_message(message: telebot.types.Message):
    """Обработчик команды /help"""
    help_text = (
        "Доступные команды:\n"
        "/start - запуск бота\n"
        "/help - помощь\n"
        "/choose_model - выбор модели (старый метод)\n"
        "/quick_model - быстрый выбор модели через кнопки\n"
        "/current_model - показать текущую выбранную модель\n"
        "/clear_keyboard - удалить клавиатуру, если она застряла\n\n"
        f"Для ChatGPT доступны модели: {', '.join(AVAILABLE_CHATGPT_MODELS)}\n"
        f"Для Claude доступны модели: {', '.join(AVAILABLE_CLAUDE_MODELS)}\n"
        f"Для DeepSeek доступны модели: {', '.join(AVAILABLE_DEEPSEEK_MODELS)}\n"
        "По умолчанию используется DeepSeek Reasoner.\n"
        "Просто отправь сообщение, и я отвечу, используя выбранную модель."
    )
    # Удаляем клавиатуру при показе помощи
    bot.send_message(message.chat.id, help_text, reply_markup=telebot.types.ReplyKeyboardRemove())

@bot.message_handler(commands=['current_model'])
def show_current_model(message: telebot.types.Message):
    """Показать текущую выбранную модель"""
    user_id = message.from_user.id
    current_provider = user_data.get_provider(user_id)
    current_model = user_data.get_model(user_id)
    
    # Создаем кнопку для быстрой смены модели
    markup = telebot.types.InlineKeyboardMarkup()
    change_model_btn = telebot.types.InlineKeyboardButton(
        "Сменить модель", 
        callback_data="select_provider"
    )
    markup.add(change_model_btn)
    
    # Удаляем реплай-клавиатуру и показываем инлайн кнопки
    bot.send_message(
        message.chat.id, 
        f"🤖 *Текущие настройки*\n\n*Провайдер*: {current_provider}\n*Модель*: {current_model}",
        parse_mode="Markdown",
        reply_markup=markup
    )

@bot.message_handler(commands=['choose_model'])
def choose_model(message: telebot.types.Message):
    """Обработчик команды /choose_model"""
    user_id = message.from_user.id
    # Убираем существующую клавиатуру перед показом новой
    remove_markup = telebot.types.ReplyKeyboardRemove()
    bot.send_message(message.chat.id, "Подготовка меню выбора...", reply_markup=remove_markup)
    
    # Теперь показываем новую клавиатуру
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    markup.add('DeepSeek', 'ChatGPT', 'Claude')
    bot.send_message(message.chat.id, "Сначала выбери провайдера:", reply_markup=markup)
    user_data.set_state(user_id, UserState.CHOOSING_PROVIDER)

# Обработчики callback-запросов
@bot.callback_query_handler(func=lambda call: call.data == "select_provider")
def callback_select_provider(call):
    """Обработчик выбора провайдера через инлайн-кнопки"""
    markup = create_provider_keyboard()
    bot.edit_message_text(
        "Выберите провайдера AI:",
        call.message.chat.id,
        call.message.message_id,
        reply_markup=markup
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith("provider_"))
def callback_provider_selected(call):
    """Обработчик выбора конкретного провайдера"""
    provider = call.data.split("_")[1]  # Извлекаем имя провайдера
    user_id = call.from_user.id
    user_data.set_provider(user_id, provider)
    
    # Показываем клавиатуру для выбора модели
    markup = create_model_keyboard(provider)
    bot.edit_message_text(
        f"Выбран провайдер: {provider}\nТеперь выберите модель:",
        call.message.chat.id,
        call.message.message_id,
        reply_markup=markup
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith(("deepseek_model_", "chatgpt_model_", "claude_model_")))
def callback_model_selected(call):
    """Обработчик выбора конкретной модели"""
    parts = call.data.split("_")
    provider_prefix = parts[0]  # deepseek, chatgpt или claude
    model_name = "_".join(parts[2:])  # Название модели (может содержать символы "_")
    
    user_id = call.from_user.id
    user_data.set_model(user_id, model_name)
    
    # Убираем клавиатуру и отображаем подтверждение
    provider = user_data.get_provider(user_id)
    
    # Создаем кнопку для повторного выбора
    markup = telebot.types.InlineKeyboardMarkup()
    change_model_btn = telebot.types.InlineKeyboardButton(
        "Выбрать другую модель", 
        callback_data="select_provider"
    )
    markup.add(change_model_btn)
    
    bot.edit_message_text(
        f"✅ *Настройки обновлены*\n\n*Провайдер*: {provider}\n*Модель*: {model_name}",
        call.message.chat.id,
        call.message.message_id,
        parse_mode="Markdown",
        reply_markup=markup
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith("quick_model_"))
def callback_quick_model_selected(call):
    """Обработчик быстрого выбора модели"""
    parts = call.data.split("_")
    provider = parts[2]  # DeepSeek, ChatGPT или Claude
    model_name = "_".join(parts[3:])  # Название модели
    
    user_id = call.from_user.id
    user_data.set_provider(user_id, provider)
    user_data.set_model(user_id, model_name)
    
    # Создаем кнопку для повторного выбора
    markup = telebot.types.InlineKeyboardMarkup()
    change_model_btn = telebot.types.InlineKeyboardButton(
        "Выбрать другую модель", 
        callback_data="select_provider"
    )
    markup.add(change_model_btn)
    
    bot.edit_message_text(
        f"✅ *Модель успешно изменена*\n\n*Провайдер*: {provider}\n*Модель*: {model_name}",
        call.message.chat.id,
        call.message.message_id,
        parse_mode="Markdown",
        reply_markup=markup
    )

# Обработчики выбора модели (для обратной совместимости)
@bot.message_handler(func=lambda msg: msg.text in ['DeepSeek', 'ChatGPT', 'Claude'])
def handle_provider_choice(message: telebot.types.Message):
    """Обработчик выбора провайдера"""
    user_id = message.from_user.id
    
    if user_data.get_state(user_id) != UserState.CHOOSING_PROVIDER:
        # Удаляем клавиатуру и информируем пользователя
        bot.send_message(message.chat.id, "Чтобы выбрать провайдера, используйте команду /choose_model", 
                        reply_markup=telebot.types.ReplyKeyboardRemove())
        return
        
    user_data.set_provider(user_id, message.text)
    
    if message.text == 'ChatGPT':
        markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        for model in AVAILABLE_CHATGPT_MODELS:
            markup.add(model)
        bot.send_message(message.chat.id, "Теперь выбери модель ChatGPT:", reply_markup=markup)
        user_data.set_state(user_id, UserState.CHOOSING_MODEL)
    elif message.text == 'Claude':
        markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        for model in AVAILABLE_CLAUDE_MODELS:
            markup.add(model)
        bot.send_message(message.chat.id, "Теперь выбери модель Claude:", reply_markup=markup)
        user_data.set_state(user_id, UserState.CHOOSING_CLAUDE_MODEL)
    else:  # DeepSeek
        markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        for model in AVAILABLE_DEEPSEEK_MODELS:
            markup.add(model)
        bot.send_message(message.chat.id, "Теперь выбери модель DeepSeek:", reply_markup=markup)
        user_data.set_state(user_id, UserState.CHOOSING_DEEPSEEK_MODEL)

@bot.message_handler(func=lambda msg: msg.text in AVAILABLE_CHATGPT_MODELS)
def handle_model_choice(message: telebot.types.Message):
    """Обработчик выбора модели ChatGPT"""
    user_id = message.from_user.id
    
    if user_data.get_state(user_id) != UserState.CHOOSING_MODEL:
        # Удаляем клавиатуру если пользователь в неправильном состоянии
        bot.send_message(message.chat.id, "Для выбора модели сначала выберите провайдера через /choose_model", 
                        reply_markup=telebot.types.ReplyKeyboardRemove())
        return
        
    user_data.set_model(user_id, message.text)
    # Явно удаляем клавиатуру после выбора
    bot.send_message(
        message.chat.id, 
        f"Выбрана модель {message.text}!", 
        reply_markup=telebot.types.ReplyKeyboardRemove()
    )
    user_data.reset_state(user_id)

@bot.message_handler(func=lambda msg: msg.text in AVAILABLE_CLAUDE_MODELS)
def handle_claude_model_choice(message: telebot.types.Message):
    """Обработчик выбора модели Claude"""
    user_id = message.from_user.id
    
    if user_data.get_state(user_id) != UserState.CHOOSING_CLAUDE_MODEL:
        # Удаляем клавиатуру если пользователь в неправильном состоянии
        bot.send_message(message.chat.id, "Для выбора модели сначала выберите провайдера через /choose_model", 
                        reply_markup=telebot.types.ReplyKeyboardRemove())
        return
        
    user_data.set_model(user_id, message.text)
    # Явно удаляем клавиатуру после выбора
    bot.send_message(
        message.chat.id, 
        f"Выбрана модель {message.text}!", 
        reply_markup=telebot.types.ReplyKeyboardRemove()
    )
    user_data.reset_state(user_id)

@bot.message_handler(func=lambda msg: msg.text in AVAILABLE_DEEPSEEK_MODELS)
def handle_deepseek_model_choice(message: telebot.types.Message):
    """Обработчик выбора модели DeepSeek"""
    user_id = message.from_user.id
    
    if user_data.get_state(user_id) != UserState.CHOOSING_DEEPSEEK_MODEL:
        # Удаляем клавиатуру если пользователь в неправильном состоянии
        bot.send_message(message.chat.id, "Для выбора модели сначала выберите провайдера через /choose_model", 
                        reply_markup=telebot.types.ReplyKeyboardRemove())
        return
        
    user_data.set_model(user_id, message.text)
    # Явно удаляем клавиатуру после выбора
    bot.send_message(
        message.chat.id, 
        f"Выбрана модель {message.text}!", 
        reply_markup=telebot.types.ReplyKeyboardRemove()
    )
    user_data.reset_state(user_id)

# Основной обработчик сообщений
@bot.message_handler(func=lambda msg: True)
def reply_message(message: telebot.types.Message):
    """Обработчик всех остальных сообщений"""
    try:
        user_id = message.from_user.id
        user_text = message.text
        
        # Если пользователь в процессе выбора модели, игнорируем обычные сообщения
        if user_data.get_state(user_id) != UserState.NORMAL:
            # Удаляем клавиатуру и просим завершить выбор
            bot.send_message(message.chat.id, "Пожалуйста, сначала заверши выбор модели через меню или используй /clear_keyboard для сброса.", 
                            reply_markup=telebot.types.ReplyKeyboardRemove())
            return

        # Получаем выбранного провайдера и модель
        provider = user_data.get_provider(user_id)
        model_choice = user_data.get_model(user_id)
        
        # Отправляем "печатает..." статус
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Выполняем запрос к соответствующему API
        if provider == 'DeepSeek':
            reply = query_deepseek(user_text, model_choice)
        elif provider == 'Claude':
            reply = query_claude(user_text, model_choice)
        else:  # OpenAI/ChatGPT
            reply = query_chatgpt(user_text, model_choice)

        # Разбиваем длинные сообщения
        if len(reply) > 4096:
            for i in range(0, len(reply), 4096):
                bot.send_message(message.chat.id, reply[i:i+4096], reply_markup=telebot.types.ReplyKeyboardRemove())
        else:
            bot.send_message(message.chat.id, reply, reply_markup=telebot.types.ReplyKeyboardRemove())
            
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}", exc_info=True)
        # Удаляем клавиатуру в случае ошибки
        bot.send_message(message.chat.id, "Произошла ошибка. Попробуйте позже.", 
                        reply_markup=telebot.types.ReplyKeyboardRemove())

# Функция для повторных попыток при ошибках сети или сервера
def retry_on_error(func, max_retries=3, backoff_factor=2):
    """Обертка для повторных попыток с экспоненциальной задержкой"""
    def wrapper(*args, **kwargs):
        import time
        retries = 0
        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                wait_time = backoff_factor ** retries
                logger.warning(f"Attempt {retries+1} failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
        # Последняя попытка без обработки исключения
        return func(*args, **kwargs)
    return wrapper

# Сохраняем оригинальные функции
original_query_deepseek = query_deepseek
original_query_chatgpt = query_chatgpt
original_query_claude = query_claude

# Оборачиваем функции API для повторных попыток
query_deepseek = retry_on_error(original_query_deepseek)
query_chatgpt = retry_on_error(original_query_chatgpt)
query_claude = retry_on_error(original_query_claude)

if __name__ == "__main__":
    logger.info("Бот запущен с настройками по умолчанию: провайдер DeepSeek, модель deepseek-reasoner")
    try:
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}", exc_info=True)