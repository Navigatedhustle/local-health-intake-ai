import ollama
import json
from pydantic import BaseModel, Field
from typing import Optional

MODEL = "mistral-nemo"

SYSTEM_PROMPT = """You are a clinical intake assistant helping gather patient information before a doctor's visit.
Your job is to have a calm, professional conversation with the patient to understand their symptoms.

Follow this flow:
1. Ask about their chief complaint (main reason for visit)
2. Ask how long they have had the symptom
3. Ask them to rate their pain or discomfort on a scale of 1 to 10
4. Ask if there are any other symptoms accompanying the main complaint
5. Ask if they have any known allergies or current medications

Keep responses short, clear, and empathetic. One question at a time.
Never diagnose. Never prescribe. You are only gathering information.

IMPORTANT: You are running 100% locally on this device. No data is transmitted to any server.
This conversation is HIPAA-safe by design.

Once you have gathered enough information (at least chief complaint, duration, and severity),
the user can click the Generate Report button to produce a structured intake summary."""


EXTRACTION_PROMPT = """Based on the following patient intake conversation, extract the structured information.

Conversation:
{conversation}

Return ONLY a valid JSON object with exactly these fields:
{{
  "chief_complaint": "the main reason for the visit in the patient's own words",
  "duration": "how long the patient has had the symptom",
  "severity_score": a number from 1 to 10 (use 0 if not mentioned),
  "additional_symptoms": ["list", "of", "other", "symptoms"] or empty list if none,
  "allergies": "any mentioned allergies or 'None reported'",
  "current_medications": "any mentioned medications or 'None reported'",
  "red_flags": ["list any concerning symptoms like chest pain, difficulty breathing, severe pain"] or empty list,
  "suggested_urgency": "one of: Routine, Urgent, or Emergency",
  "urgency_reasoning": "one sentence explaining why you assigned that urgency level"
}}

Return only the JSON. No explanation. No markdown. No code blocks. Just the raw JSON object."""


class IntakeReport(BaseModel):
    chief_complaint: str = Field(description="Main reason for the visit")
    duration: str = Field(description="How long the symptom has been present")
    severity_score: int = Field(description="Pain or discomfort score 1-10", ge=0, le=10)
    additional_symptoms: list[str] = Field(default_factory=list)
    allergies: str = Field(default="None reported")
    current_medications: str = Field(default="None reported")
    red_flags: list[str] = Field(default_factory=list)
    suggested_urgency: str = Field(description="Routine, Urgent, or Emergency")
    urgency_reasoning: str = Field(description="Why this urgency level was assigned")


def get_ai_response(conversation_history: list[dict]) -> str:
    """Send conversation history to local model and get next response."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
    
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={
            "temperature": 0.3,
            "num_predict": 300,
        }
    )
    
    return response["message"]["content"]


def extract_structured_data(conversation_history: list[dict]) -> Optional[IntakeReport]:
    """Extract structured intake data from the full conversation."""
    conversation_text = ""
    for msg in conversation_history:
        role = "Patient" if msg["role"] == "user" else "Intake Assistant"
        conversation_text += f"{role}: {msg['content']}\n\n"
    
    prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)
    
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,
            "num_predict": 600,
        }
    )
    
    raw = response["message"]["content"].strip()
    
    raw = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(raw)
        
        if isinstance(data.get("severity_score"), str):
            try:
                data["severity_score"] = int(data["severity_score"])
            except ValueError:
                data["severity_score"] = 0
        
        report = IntakeReport(**data)
        return report
    except (json.JSONDecodeError, Exception) as e:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                json_str = raw[start:end]
                data = json.loads(json_str)
                report = IntakeReport(**data)
                return report
        except Exception:
            pass
        return None


def get_opening_message() -> str:
    """Get the AI's opening greeting."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Hello, I need to check in for my appointment."}
        ],
        options={
            "temperature": 0.3,
            "num_predict": 200,
        }
    )
    return response["message"]["content"]