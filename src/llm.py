from langchain_google_genai import ChatGoogleGenerativeAI
from .config import GOOGLE_API_KEY
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

def get_llm():
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        return llm
    except Exception as e:
        raise

llm = get_llm()