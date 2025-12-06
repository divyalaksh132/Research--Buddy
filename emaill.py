# file: emaill.py
from trycourier import Courier

def send_email(email, ans, auth):
    client = Courier(auth_token=auth)
    client.send_message(
        message={
            "to": {"email": email},
            "content": {"title": "Answer from Research Buddy", "body": ans},
            "routing": {"method": "single", "channels": ["email"]},
        }
    )
    return True
