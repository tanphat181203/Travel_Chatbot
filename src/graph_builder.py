from typing import Sequence, Optional, List
from datetime import date
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

from src.graph_state import GraphState
from src.llm import llm
from src.prompts import response_gen_prompt, routing_prompt
from src.tools import extract_entities_tool, search_tours_tool, fetch_locations_tool
from src.database import get_available_locations, get_tour_by_id

def fetch_context(state: GraphState) -> GraphState:
    current_date_str = date.today().strftime('%Y-%m-%d')
    locations = fetch_locations_tool()
    if not locations:
        locations = []

    user_query = ""
    if state.get("messages"):
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            user_query = last_message.content

    return {
        **state,
        "current_date": current_date_str,
        "available_locations": locations,
        "user_query": user_query
    }

def route_query(state: GraphState) -> GraphState:
    user_query = state.get("user_query", "")
    messages = state.get("messages", [])
    chat_history = "\n".join([f"{m.type}: {m.content}" for m in messages[:-1]])

    if not user_query:
        return {**state, "routing_decision": "error_state"}

    prompt = routing_prompt.format(chat_history=chat_history, user_query=user_query)

    try:
        ai_message = llm.invoke(prompt)
        route = ai_message.content.strip().lower()

        valid_routes = ["search", "respond", "error_state"]
        if route not in valid_routes:
            route = "respond"

        return {**state, "routing_decision": route}
    except Exception as e:
        return {**state, "routing_decision": "respond", "error": str(e)}

def get_routing_decision(state: GraphState) -> str:
    decision = state.get("routing_decision", "respond")
    return decision

def extract_entities(state: GraphState) -> GraphState:
    user_query = state["user_query"]
    current_date_str = state["current_date"]

    entities = extract_entities_tool(user_query, current_date_str)

    if entities and isinstance(entities, dict) and "error" in entities:
        return {**state, "error": entities["error"], "extracted_entities": None}

    return {**state, "extracted_entities": entities, "error": None}

def search_tours(state: GraphState) -> GraphState:
    entities = state.get("extracted_entities")
    if not entities or "error" in entities:
        return {**state, "search_results": []}

    try:
        search_results = search_tours_tool(entities)

        if search_results is None:
            search_results = []
        return {**state, "search_results": search_results}
    except Exception as e:
        return {**state, "search_results": [], "error": str(e)}

def generate_response(state: GraphState) -> GraphState:
    user_query = state["user_query"].lower()
    messages = state.get("messages", [])
    search_results = state.get("search_results", [])

    if search_results is None:
        search_results = []

    error = state.get("error")

    itinerary_keywords = ["lịch trình", "hành trình", "lộ trình", "chương trình du lịch", "kế hoạch du lịch"]
    booking_keywords = ["đặt tour", "book tour", "đặt chỗ", "đăng ký tour", "mua tour", "đặt vé", "reserve", "tôi muốn đi", "tôi muốn đặt"]

    is_ask_itinerary = any(kw in user_query for kw in itinerary_keywords)
    is_booking_request = any(kw in user_query for kw in booking_keywords)

    tour_name = None
    tour_id = None
    itinerary_text = None
    tour_index = None

    if is_ask_itinerary:
        import re
        last_tour_name = None
        last_tour_id = None
        last_tour_index = None

        match3 = re.search(r"tour *(thứ|số)? *(\d+|một|hai|ba|bốn|năm|đầu tiên)", user_query, re.IGNORECASE)
        if match3:
            val = match3.group(2).strip().lower()
            num_map = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "đầu tiên": 1}
            try:
                idx = int(val)
            except:
                idx = num_map.get(val, None)
            if idx:
                last_tour_index = idx - 1

        if last_tour_index is None and last_tour_id is None and last_tour_name is None:
            for m in reversed(messages[:-1]):
                if hasattr(m, 'content') and m.content:
                    match = re.search(r"tour ([^\(\n]+) \(ID: (\d+)\)", m.content, re.IGNORECASE)
                    if match:
                        last_tour_name = match.group(1).strip()
                        last_tour_id = int(match.group(2))
                        break

                    id_match = re.search(r"\(ID: (\d+)\)", m.content, re.IGNORECASE)
                    if id_match:
                        last_tour_id = int(id_match.group(1))
                        break

                    match2 = re.search(r"tour ([^\(\n]+)", m.content, re.IGNORECASE)
                    if match2:
                        last_tour_name = match2.group(1).strip()
                        break

                    match3_hist = re.search(r"tour *(thứ|số)? *(\d+|một|hai|ba|bốn|năm|đầu tiên)", m.content, re.IGNORECASE)
                    if match3_hist:
                        val_hist = match3_hist.group(2).strip().lower()
                        num_map_hist = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "đầu tiên": 1}
                        try:
                            idx_hist = int(val_hist)
                        except:
                            idx_hist = num_map_hist.get(val_hist, None)
                        if idx_hist:
                            last_tour_index = idx_hist - 1
                            break

        found_tour = None

        if not search_results and last_tour_id:
            try:
                from src.database import get_tour_by_id
                db_tour = get_tour_by_id(last_tour_id)
                if db_tour:
                    from src.tools import format_itineraries
                    search_results = format_itineraries([db_tour])
            except Exception as e:
                pass

        if search_results:
            if last_tour_index is not None and 0 <= last_tour_index < len(search_results):
                found_tour = search_results[last_tour_index]
            elif last_tour_id:
                for t in search_results:
                    if str(t.get("tour_id")) == str(last_tour_id):
                        found_tour = t
                        break
            elif last_tour_name:
                for t in search_results:
                    if last_tour_name.lower() in t.get("title", "").lower():
                        found_tour = t
                        break

        if not found_tour and search_results and (
            "tour thứ 1" in user_query.lower() or
            "tour đầu tiên" in user_query.lower() or
            "tour thứ nhất" in user_query.lower()):
            found_tour = search_results[0]

        if not found_tour and search_results:
            found_tour = search_results[0]

        if found_tour:
            if found_tour.get("itinerary"):
                itinerary_text = f"Lịch trình chi tiết của tour {found_tour.get('title', '')} (ID: {found_tour.get('tour_id', '')}):\n\n{found_tour.get('itinerary')}"
            else:
                itinerary_text = f"Xin lỗi, hiện tại tôi chưa có thông tin chi tiết về lịch trình của tour {found_tour.get('title', '')} (ID: {found_tour.get('tour_id', '')})."
        else:
            if last_tour_id:
                itinerary_text = f"Xin lỗi, tôi không tìm thấy thông tin chi tiết cho tour có ID {last_tour_id}. Vui lòng thử lại sau."
            else:
                itinerary_text = "Xin lỗi, tôi không tìm thấy thông tin lịch trình cho tour bạn quan tâm. Bạn có thể cung cấp tên tour hoặc ID tour không?"

        final_response_content = itinerary_text
        updated_messages = list(messages) + [AIMessage(content=final_response_content)]
        return {**state, "messages": updated_messages, "final_response": final_response_content, "error": None}

    if error:
        search_results_str = f"An error occurred in a previous step: {error}"
    elif search_results:
        results_summary = []
        for i, tour in enumerate(search_results[:5]):
            price_adult = f"{tour['price_adult']:,.0f} VND" if tour.get('price_adult') else "N/A"
            price_child_120 = f"{tour['price_child_120_140']:,.0f} VND" if tour.get('price_child_120_140') else "N/A"
            price_child_100 = f"{tour['price_child_100_120']:,.0f} VND" if tour.get('price_child_100_120') else "N/A"

            promo_info = ""
            if tour.get('promotion_id'):
                discount_str = f"{tour['promotion_discount']}%" if tour.get('promotion_type') == 'percent' else f"{tour['promotion_discount']:,.0f} VND"
                promo_info = f" (KM: {tour['promotion_name']} - Giảm {discount_str})"

            summary = (
                f"{i+1}. Tour: {tour.get('title', 'N/A')} (ID: {tour.get('tour_id')})\n"
                f"   Khởi hành: {tour.get('start_date', 'N/A')}\n"
                f"   Thời gian: {tour.get('duration', 'N/A')}\n"
                f"   Giá người lớn: {price_adult}{promo_info}\n"
                f"   Giá trẻ em (1m2-1m4): {price_child_120}\n"
                f"   Giá trẻ em (1m-1m2): {price_child_100}"
            )
            results_summary.append(summary)
        search_results_str = "\n".join(results_summary)
        if len(search_results) > 5:
            search_results_str += f"\n... và {len(search_results) - 5} kết quả khác."
        search_results_str += "\n\nLưu ý: Giá vé này chưa bao gồm vé cho em bé dưới 100cm (thường được miễn phí vé dịch vụ tour, chỉ tính vé máy bay/tàu nếu có và chi phí phát sinh nếu sử dụng dịch vụ riêng)."
    elif state.get("extracted_entities") and not search_results:
        search_results_str = "Xin lỗi, tôi không tìm thấy tour nào phù hợp với yêu cầu của bạn."
    else:
        search_results_str = "Không có thông tin tìm kiếm liên quan."

    chat_history_messages = []
    chat_history = ""
    if messages:
        history_to_include = messages[:-1]
        if history_to_include:
            chat_history = "\n".join([f"{m.type}: {m.content}" for m in history_to_include])
        chat_history_messages.extend(history_to_include)

    prompt = response_gen_prompt.format_messages(
        chat_history_messages=chat_history_messages,
        chat_history=chat_history,
        search_results=search_results_str,
        user_query=user_query
    )

    try:
        ai_response = llm.invoke(prompt)
        final_response_content = ai_response.content
        updated_messages = list(messages) + [AIMessage(content=final_response_content)]
        return {**state, "messages": updated_messages, "final_response": final_response_content, "error": None}
    except Exception as e:
        error_message = "Xin lỗi, tôi gặp sự cố khi tạo câu trả lời."
        updated_messages = list(messages) + [AIMessage(content=error_message)]
        return {**state, "messages": updated_messages, "final_response": error_message, "error": str(e)}

def handle_error(state: GraphState) -> GraphState:
    error = state.get("error", "Lỗi không xác định.")
    error_message = f"Xin lỗi, đã có lỗi xảy ra: {error}. Vui lòng thử lại hoặc hỏi khác đi."
    messages = list(state.get("messages", [])) + [AIMessage(content=error_message)]
    return {**state, "messages": messages, "final_response": error_message}

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("fetch_context", fetch_context)
    workflow.add_node("route_query", route_query)
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("search_tours", search_tours)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("handle_error", handle_error)

    workflow.set_entry_point("fetch_context")

    workflow.add_edge("fetch_context", "route_query")

    workflow.add_conditional_edges(
        "route_query",
        get_routing_decision,
        {
            "search": "extract_entities",
            "respond": "generate_response",
            "error_state": "handle_error",
        }
    )

    workflow.add_edge("extract_entities", "search_tours")
    workflow.add_edge("search_tours", "generate_response")

    workflow.add_edge("generate_response", END)
    workflow.add_edge("handle_error", END)

    app = workflow.compile()
    return app

graph_app = build_graph()