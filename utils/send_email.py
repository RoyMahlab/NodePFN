from dotenv import load_dotenv
import os
load_dotenv(override=True)  # Load environment variables from .env file
import smtplib
from email.message import EmailMessage

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = "roy.mahlab@gmail.com"
SENDER_PASSWORD = os.getenv("GOOGLE_ACCOUNT_PASSWORD")  # Use an App Password, not your normal password
RECIPIENT_EMAIL = "roy.mahlab@gmail.com"

# Create the email
msg = EmailMessage()
msg["Subject"] = "Test Email"
msg["From"] = SENDER_EMAIL
msg["To"] = RECIPIENT_EMAIL

def send_email(subject: str, body: str):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg.set_content(body)

    # Send the email
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.starttls()  # Encrypt the connection
        smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
        smtp.send_message(msg)

    print("Email sent successfully!")
