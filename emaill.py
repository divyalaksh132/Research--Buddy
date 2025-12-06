from trycourier import Courier

def send_email(email, answer, auth):
    client = Courier(auth_token=auth)
    client.send_message(
        message={
            "to": {"email": email},
            "content": {
                "title": "Your Research Buddy Answer",
                "body": answer
            },
            "routing": {"method": "single", "channels": ["email"]}
        }
    )
    return True
