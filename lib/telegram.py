import requests


class Telegram:
    def __init__(self, conf):
        self.conf = conf

    def notify(self, message):
        if not self.conf.telegram.enabled:
            return

        url = f"https://api.telegram.org/bot{self.conf.telegram.token}"
        params = {"chat_id": self.conf.telegram.id, "text": message}

        try:
            requests.get(url + "/sendMessage", params=params)
        except:
            print("Failed to send notification.")
