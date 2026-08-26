from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

import requests
import os


class ChatbotAgentState(TypedDict):
    user_input: str
    bot_response: str
    context: str


SAIS_APP_URL = os.getenv("SAIS_APP_URL")
SAIS_TOKEN = os.getenv("SAIS_TOKEN")
SAIS_MODEL = os.getenv("SAIS_MODEL", "gpt-4.1")

if not SAIS_APP_URL or not SAIS_TOKEN:
    raise SystemExit("Error: SAIS_APP_URL and SAIS_TOKEN environment variables must be set.")

PROXY_HOST = os.getenv("PROXY_HOST")
PROXY_PORT = os.getenv("PROXY_PORT")
PROXY_ENABLED = os.getenv("PROXY_ENABLED", "false").lower() == "true"
SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() == "true"


def generate_answer(context: str, query: str) -> str:
    response = requests.post(
        f"{SAIS_APP_URL}/v1/responses",
        headers={
            "Authorization": f"Bearer {SAIS_TOKEN}",
            "Content-Type": "application/json",
            "ApplicationType": "BRProduct",
        },
        json={
            "model": SAIS_MODEL,
            "instructions": "You are an AI technical support assistant.\n\nAnswer the user's question using ONLY the provided knowledge base context.\n\nRules:\n\n1. Use only information provided in the context.\n2. Do not invent technical details.\n3. Do not invent commands, error codes, configurations, or troubleshooting steps.\n4. If the answer cannot be determined from the context, clearly state that the information is not available in the knowledge base.\n5. Provide a concise but technically useful answer.\n6. Any factual claim based on the knowledge base MUST include a citation.\n7. Use the citation format [1], [2], [3], etc.\n8. Only cite sources that actually support the claim.\n9. Do not create citation numbers that do not exist.\n10. Do not cite irrelevant documents.\n11. Preserve technical identifiers exactly.",
            "input": context + "\n\n" + query,
        },
        verify=SSL_VERIFY,
        proxies={
            "http": f"http://{PROXY_HOST}:{PROXY_PORT}" if PROXY_ENABLED else None,
            "https": f"http://{PROXY_HOST}:{PROXY_PORT}" if PROXY_ENABLED else None,
        }
    )
    if response.status_code == 200:
        response_data = response.json()
        try:
            outputs = response_data["body"]["output"]
            texts = []
            for item in outputs:
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            texts.append(content.get("text", ""))
            if texts:
                return "\n".join(texts)
            return "Error: No output text found in the response."
        except KeyError:
            return "Error: Unexpected response format."
    else:
        return f"Error: Request failed with status code {response.status_code}"


def process(state: ChatbotAgentState) -> ChatbotAgentState:
    context = state.get("context", "")
    user_input = state.get("user_input", "")
    bot_response = generate_answer(context, user_input)
    return {
        "user_input": user_input,
        "bot_response": bot_response,
        "context": context
    }

graph = StateGraph(ChatbotAgentState)
graph.add_node("process", process)
graph.set_entry_point("process")
graph.add_edge("process", END)

app = graph.compile()

while True:
    user_input = input("Enter your question (type 'exit' to quit): ")
    if user_input.strip().lower() == "exit":
        break
    result = app.invoke({
        "user_input": user_input,
        "bot_response": "",
        "context": "",
    })
    print("Bot:", result["bot_response"])


    
