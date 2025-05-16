from datetime import date
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from typing import List, Sequence

from src.graph_builder import graph_app
from src.graph_state import GraphState
from src.database import conn_pool

def run_chatbot():
    try:
        if conn_pool is None:
            print("\nLỗi: Không thể kết nối Database. Chatbot không thể hoạt động.")
            return

        from src.tools import fetch_locations_tool
        initial_locations = fetch_locations_tool()
        if initial_locations is None:
            print("\nLỗi: Không thể truy vấn dữ liệu từ Database. Chatbot không thể hoạt động.")
            return
        print(f"\nChatbot đã sẵn sàng! Các địa điểm hỗ trợ: {', '.join(initial_locations[:10])}{'...' if len(initial_locations) > 10 else ''}")

    except Exception as e:
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
            print(f"Chatbot: Đã xảy ra lỗi: {e}. Vui lòng thử lại.")

if __name__ == "__main__":
    run_chatbot()