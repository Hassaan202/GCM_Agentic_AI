import smtplib

def send_email(subject, body, sender_email, app_password, receiver_email):
    if sender_email is None and app_password is None:
        return "Sender Credentials not configured! Email not sent."

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        text = f"Subject: {subject}\n\n{body}\nTesting Only!"
        server.sendmail(sender_email, receiver_email, text)
        return "Email sent successfully!"
    except Exception as e:
        print(e)
        return "Something went wrong!"