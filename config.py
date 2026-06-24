from dotenv import load_dotenv
import os

load_dotenv()

OWM_API_KEY = os.environ["OWM_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]