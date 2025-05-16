import logging
from datetime import date
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from typing import List, Sequence

from src.graph_builder import graph_app
from src.graph_state import GraphState
from src.database import conn_pool


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_chatbot():
    logger.info("Starting chatbot...")
    logger.info("Kiểm tra kết nối Database...")
    try:

        if conn_pool is None:
             logger.error("Lỗi: Không thể khởi tạo Database connection pool. Kiểm tra log lỗi trong src/database.py và file .env.")
             print("\nLỗi: Không thể kết nối Database. Chatbot không thể hoạt động.")
             return


        from src.tools import fetch_locations_tool
        initial_locations = fetch_locations_tool()
        if initial_locations is None:
            logger.error("Lỗi: Không thể kết nối hoặc truy vấn Database sau khi pool khởi tạo. Vui lòng kiểm tra log và cấu hình.")
            print("\nLỗi: Không thể truy vấn dữ liệu từ Database. Chatbot không thể hoạt động.")
            return
        logger.info(f"Kết nối Database và truy vấn thành công. Tìm thấy {len(initial_locations)} địa điểm.")
        print(f"\nChatbot đã sẵn sàng! Các địa điểm hỗ trợ: {', '.join(initial_locations[:10])}{'...' if len(initial_locations) > 10 else ''}")

    except Exception as e:
         logger.error(f"Lỗi khởi tạo chatbot hoặc kiểm tra Database: {e}", exc_info=True)
         print(f"\nLỗi khởi tạo: {e}")
         return

    conversation_history: List[BaseMessage] = []

    print("\n--- Bắt đầu trò chuyện (gõ 'quit' để thoát) ---")


    while True:
        try:
            user_input = input("Bạn: ")
            if user_input.lower() == 'quit':
                print("Chatbot: Tạm biệt!")
                break
            if not user_input.strip():
                continue

            conversation_history.append(HumanMessage(content=user_input))

            graph_input: GraphState = {
                "messages": conversation_history,
                "user_query": None, "current_date": None, "available_locations": None,
                "extracted_entities": None, "search_results": None,
                "final_response": None, "error": None,
                "routing_decision": None,
            }

            final_state = graph_app.invoke(graph_input)

            conversation_history = list(final_state.get("messages", conversation_history))
            response = final_state.get("final_response", "Xin lỗi, tôi không thể xử lý yêu cầu này.")

            print(f"Chatbot: {response}")

        except KeyboardInterrupt:
            print("\nChatbot: Tạm biệt!")
            break
        except Exception as e:
            logger.exception("Lỗi trong vòng lặp chat:")
            print(f"Chatbot: Đã xảy ra lỗi: {e}. Vui lòng thử lại.")


if __name__ == "__main__":
    run_chatbot()