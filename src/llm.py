from langchain_google_genai import ChatGoogleGenerativeAI
from .config import GOOGLE_API_KEY
import logging

logger = logging.getLogger(__name__)

def get_llm():
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        logger.info("Gemini LLM initialized successfully.")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {e}")
        raise

llm = get_llm()