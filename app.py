import os
import json
import logging
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr

# Load environment variables
load_dotenv(override=True)

# Suppress noisy logs in production
logging.getLogger().setLevel(logging.WARNING)

# Optional debug logging
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def push(message):
    if DEBUG:
        print(f"[DEBUG] Push: {message}")
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": message,
        }
    )

# Tool function: record user email
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

# Tool function: log unknown questions
def record_unknown_question(question):
    push(f"Recording unknown question: {question}")
    return {"recorded": "ok"}

# Tool schema definitions
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The user's email address"},
            "name": {"type": "string", "description": "The user's name (optional)"},
            "notes": {"type": "string", "description": "Extra context or message (optional)"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question text"}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

class Me:
    def __init__(self):
        self.name = "Moamen"

        groq_api_key = os.getenv("GROQ_API_KEY")
        self.client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
        self.model_name = "llama-3.3-70b-versatile"

        # Load LinkedIn data
        reader = PdfReader("me/Profile.pdf")  # <-- Make sure file is named exactly this
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        # Load summary
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def system_prompt(self):
        return (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. "
            f"You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. "
            f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            f"If you don't know the answer to any question, use your record_unknown_question tool to record the question. "
            f"If the user is engaging in discussion, try to steer them towards getting in touch via email; "
            f"ask for their email and record it using your record_user_details tool.\n\n"
            f"## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
            f"With this context, please chat with the user, always staying in character as {self.name}."
        )

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            if DEBUG:
                print(f"[DEBUG] Tool called: {tool_name}")
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results

    def chat(self, message, history):
        # Clean up message history for compatibility
        clean_history = []
        for msg in history:
            if isinstance(msg, dict):
                clean_msg = {"role": msg["role"], "content": msg["content"]}
                clean_history.append(clean_msg)
            else:
                clean_history.append(msg)

        messages = [{"role": "system", "content": self.system_prompt()}] + clean_history + [
            {"role": "user", "content": message}
        ]

        done = False
        while not done:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools
            )

            choice = response.choices[0]
            finish_reason = choice.finish_reason

            if finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls
                tool_results = self.handle_tool_call(tool_calls)
                messages.append(choice.message)
                messages.extend(tool_results)
            else:
                done = True

        return choice.message.content

# Gradio app entry point
if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(fn=me.chat).launch()
