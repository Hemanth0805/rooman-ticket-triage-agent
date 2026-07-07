import os
import json
import argparse
from glob import glob
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

try:
    import openai
    from openai import OpenAI
except Exception:
    openai = None
    OpenAI = None

load_dotenv()


def ensure_openai(api_key=None):
    """Ensure OpenAI is configured; can accept an api_key directly."""
    env_key = api_key or os.getenv("OPENAI_API_KEY")
    if not env_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment, .env file, or pass --api-key.")
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed. See requirements.txt")
    return OpenAI(api_key=env_key)


SYSTEM_PROMPT = (
    "You are a support-ticket triage assistant. For each ticket (subject and body), "
    "respond with a JSON object containing the following fields:\n"
    "- category: one of [Billing, Technical, Account, Feature Request, Other]\n"
    "- urgency: one of [High, Medium, Low]\n"
    "- routing_team: short team name (e.g., Billing Team, Engineering, Accounts)\n"
    "- confidence: number between 0 and 1\n"
    "- reasoning: 1-2 sentence explanation for the decision.\n"
    "Return ONLY the JSON object."
)


def classify_ticket_openai(ticket_text, model="gpt-4", api_key=None):
    client = ensure_openai(api_key=api_key)
    prompt = f"Subject: {ticket_text.get('subject','')}\nBody: {ticket_text.get('body','')}\n\nRespond in JSON as specified."
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_output_tokens=400,
    )
    # New API returns nested output structure
    text = ""
    if resp.output and len(resp.output) > 0:
        for item in resp.output:
            if item.get("type") == "message" and item.get("content"):
                contents = item["content"]
                if isinstance(contents, dict) and contents.get("text"):
                    text += contents["text"]
                elif isinstance(contents, str):
                    text += contents
    text = text.strip()
    try:
        # Some models return code fences — try to extract JSON
        if text.startswith("```"):
            text = text.strip('`')
        obj = json.loads(text)
    except Exception:
        # Fallback: try to find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            obj = json.loads(text[start:end+1])
        else:
            raise
    return obj


def classify_ticket_mock(ticket_text):
    """Simple rule-based fallback classifier for offline testing."""
    subject = (ticket_text.get("subject") or "").lower()
    body = (ticket_text.get("body") or "").lower()
    text = subject + "\n" + body
    # default values
    category = "Other"
    urgency = "Low"
    routing = "Support"
    confidence = 0.6
    reasoning = "Heuristic rules matched keywords."

    if any(k in text for k in ["billing", "invoice", "charged", "refund"]):
        category = "Billing"
        routing = "Billing Team"
        urgency = "Medium"
        confidence = 0.9
        reasoning = "Detected billing-related keywords."
    elif any(k in text for k in ["password", "log in", "2fa", "account", "access"]):
        category = "Account"
        routing = "Accounts"
        urgency = "High" if any(k in text for k in ["cannot", "blocked", "lost access"]) else "Medium"
        confidence = 0.85
        reasoning = "Account access / authentication keywords present."
    elif any(k in text for k in ["crash", "500", "error", "exception", "upload"]):
        category = "Technical"
        routing = "Engineering"
        urgency = "High" if "500" in text or "crash" in text else "Medium"
        confidence = 0.9
        reasoning = "Technical error indicators found."
    elif any(k in text for k in ["feature", "export", "api", "request"]):
        category = "Feature Request"
        routing = "Product"
        urgency = "Low"
        confidence = 0.7
        reasoning = "Language suggests a feature request."
    elif any(k in text for k in ["slow", "performance", "timeout"]):
        category = "Technical"
        routing = "Engineering"
        urgency = "Medium"
        confidence = 0.75
        reasoning = "Performance-related keywords detected."

    return {
        "category": category,
        "urgency": urgency,
        "routing_team": routing,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def load_tickets_from_path(path):
    items = []
    if os.path.isdir(path):
        for f in glob(os.path.join(path, "*.json")):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    items.extend(data)
                else:
                    items.append(data)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                items.extend(data)
            else:
                items.append(data)
    return items


def main():
    parser = argparse.ArgumentParser(description="Support Ticket Triage Agent")
    parser.add_argument("--input", default="data/tickets/sample_tickets.json", help="Input file or folder")
    parser.add_argument("--output", default="outputs/routing.csv", help="Output CSV file")
    parser.add_argument("--model", default=os.getenv("TRIAGE_MODEL", "gpt-4"), help="LLM model to use")
    parser.add_argument("--mock", action="store_true", help="Use local mock classifier instead of calling OpenAI")
    parser.add_argument("--api-key", default=None, help="One-off OpenAI API key (use locally; do not share)")
    args = parser.parse_args()

    tickets = load_tickets_from_path(args.input)
    results = []
    for t in tqdm(tickets, desc="Processing tickets"):
        try:
            if args.mock:
                res = classify_ticket_mock(t)
            else:
                res = classify_ticket_openai(t, model=args.model, api_key=args.api_key)
        except Exception as e:
            res = {"category": "Error", "urgency": "Low", "routing_team": "Support", "confidence": 0.0, "reasoning": str(e)}
        results.append({
            "id": t.get("id"),
            "subject": t.get("subject"),
            "body": t.get("body"),
            "category": res.get("category"),
            "urgency": res.get("urgency"),
            "routing_team": res.get("routing_team"),
            "confidence": res.get("confidence"),
            "reasoning": res.get("reasoning"),
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(results)} routed tickets to {args.output}")


if __name__ == "__main__":
    main()
