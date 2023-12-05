import requests as rq
import credentials
# Telegram BOT
telegram_url = credentials.telegram_url
telegram_bot_id = credentials.telegram_bot_id
telegram_chat_id = credentials.telegram_chat_id


def send_message(message):
    """Send message to the telegram channel."""
    response = rq.post(
        f'{telegram_url}/{telegram_bot_id}/sendMessage?chat_id={telegram_chat_id}&parse_mode=Markdown&text={message}')

    return response
