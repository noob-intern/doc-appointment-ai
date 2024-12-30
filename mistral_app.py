import os
import json
import requests
from flask import Flask, request, jsonify
from mistralai import Mistral
from dotenv import load_dotenv
import uuid

load_dotenv()


app = Flask(__name__)


MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
MISTRAL_MODEL = "open-mistral-7b"
FLASK_API_BASE_URL = "http://127.0.0.1:5000"


mistral_client = Mistral(api_key=MISTRAL_API_KEY)


tools = [
    {
        "type": "function",
        "function": {
            "name": "check_timeslot_availability",
            "description": "Check if a timeslot is available for a specific doctor on a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {"type": "integer", "description": "ID of the doctor."},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "time": {"type": "string", "description": "Time in HH:MM (24-hour) format."},
                },
                "required": ["doctor_id", "date", "time"],
            },
        },
    }
]


def check_timeslot_availability(doctor_id: int, date: str, time: str) -> str:
    """Call the Flask API to check if the given timeslot is available."""
    url = f"{FLASK_API_BASE_URL}/doctors/{doctor_id}/schedule/{date}"
    response = requests.get(url)
    if response.status_code == 200:
        schedule = response.json().get("schedule", [])
        for slot in schedule:
            if slot["time"] == time:
                return json.dumps({"available": slot["status"] == "available"})
        return json.dumps({"error": "Timeslot not found."})
    return json.dumps({"error": "Failed to fetch schedule from API."})


tools_to_functions = {
    "check_timeslot_availability": check_timeslot_availability,
}


def generate_tool_call_id():
    return str(uuid.uuid4())


@app.route("/chat", methods=["POST"])
def chatbot():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"error": "Message is required."}), 400

    # Step 1: Send user message and tools to Mistral
    messages = [{"role": "user", "content": user_message}]
    mistral_response = mistral_client.chat.complete(
        model=MISTRAL_MODEL, messages=messages, tools=tools, tool_choice="any"
    )

    # Step 2: Check if there are any tool calls in the Mistral response
    tool_calls = mistral_response.choices[0].message.tool_calls
    if tool_calls:
        tool_call = tool_calls[0]
        function_name = tool_call.function.name
        function_arguments = json.loads(tool_call.function.arguments)

        if function_name in tools_to_functions:
            function_result = tools_to_functions[function_name](**function_arguments)
            tool_call_id = generate_tool_call_id()

            # Append tool_call_id in assistant's tool response
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": function_name,
                                "arguments": tool_call.function.arguments,
                            },
                            "tool_call_id": tool_call_id,  # Ensure tool_call_id is added
                        }
                    ],
                }
            )

            # Include tool_call_id in the tool role response
            messages.append(
                {
                    "role": "tool",
                    "name": function_name,
                    "content": function_result,
                    "tool_call_id": tool_call_id,  # Include the ID here
                }
            )

            final_response = mistral_client.chat.complete(
                model=MISTRAL_MODEL, messages=messages
            )
            return jsonify({"response": final_response.choices[0].message.content})

    # Default response if no tool calls
    return jsonify({"response": mistral_response.choices[0].message.content})


if __name__ == "__main__":
    app.run(debug=True, port=5001)