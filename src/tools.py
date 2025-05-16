import json
import logging
from datetime import date
from .llm import llm
from .prompts import ner_prompt
from .database import search_tours_db, get_available_locations
import dateparser
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_cached_locations = None
_locations_fetched_date = None

def fetch_locations_tool():
    global _cached_locations, _locations_fetched_date
    today = date.today()
    if _cached_locations is None or _locations_fetched_date != today:
        logger.info("Fetching or refreshing available locations from DB...")
        _cached_locations = get_available_locations()
        _locations_fetched_date = today
        logger.info(f"Fetched {_cached_locations} locations.")
    return _cached_locations if _cached_locations else []

def format_itineraries(tours_array):
    for tour in tours_array:
        if isinstance(tour.get('itinerary'), list):
            itinerary_str = ""
            days = sorted(tour['itinerary'], key=lambda x: x.get('day_number', 0))

            for day in days:
                day_number = day.get('day_number', '')
                title = day.get('title', '')
                description_html = day.get('description', '')

                try:
                    soup = BeautifulSoup(description_html, 'html.parser')
                    description_text = soup.get_text(separator=' ')
                except:
                    description_text = description_html

                itinerary_str += f"Ngày {day_number}: {title}\n{description_text}\n\n"

            tour['itinerary'] = itinerary_str.strip()

    return tours_array

def extract_entities_tool(user_query: str, current_date_str: str) -> dict:
    locations = fetch_locations_tool()
    if not locations:
        logger.warning("No locations available for NER prompt.")


    prompt = ner_prompt.format(
        current_date=current_date_str,
        locations=", ".join(locations),
        question=user_query
    )
    logger.info("Running NER tool...")
    logger.debug(f"NER Prompt: {prompt}")

    try:
        from .llm import llm
        if llm is None:
            logger.error("LLM is not initialized.")
            return {"error": "LLM not available"}

        ai_message = llm.invoke(prompt)
        content = ai_message.content
        logger.debug(f"NER Raw Output: {content}")


        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()


        entities = json.loads(content)
        logger.info(f"Extracted entities: {entities}")
        return entities

    except json.JSONDecodeError as e:
        logger.error(f"NER Tool: Failed to parse JSON response: {e}. Raw content: {content}")
        try:
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                potential_json = match.group(0)
                entities = json.loads(potential_json)
                logger.warning(f"Recovered JSON: {entities}")
                return entities
            else:
                 return {"error": "Invalid JSON response from LLM", "raw_output": content}
        except Exception as inner_e:
             logger.error(f"Recovery attempt failed: {inner_e}")
             return {"error": "Invalid JSON response from LLM", "raw_output": content}

    except Exception as e:
        logger.error(f"NER Tool: An unexpected error occurred: {e}")
        return {"error": str(e)}


def search_tours_tool(entities: dict) -> list:
    logger.info(f"Running search tours tool with entities: {entities}")
    if not isinstance(entities, dict) or "error" in entities:
        logger.warning("Skipping search due to invalid or error entities.")
        return []

    if not any(key in entities for key in ['region', 'destination', 'duration', 'time', 'budget', 'number_of_people']):
         logger.info("No searchable entities provided, skipping database search.")
         return []

    try:
        search_results = search_tours_db(entities)

        if search_results is None:
            logger.warning("Database search returned None, using empty list")
            return []

        logger.info(f"Found {len(search_results)} results from database.")


        formatted_results = format_itineraries(search_results)
        logger.info(f"Formatted search results: {formatted_results}")
        return formatted_results
    except Exception as e:
        logger.error(f"Search Tours Tool: Error during database search or formatting: {e}", exc_info=True)
        return []