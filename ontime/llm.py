"""LLM framework (Section 5.4): the software computes, the AI only phrases.

Six steps for every explanation and draft message, only step 3 uses AI:
  1. Take the order's JSON record (late probability, P80 date, SHAP reasons, actions from the rule table)
  2. Mask personal data with the placeholders [CUSTOMER_NAME], [ADDRESS], [PHONE], [EMAIL]
  3. Call the LLM (low temperature, at most 8 seconds, one retry)
  4. Check the output: valid JSON, at most 3 sentences per field, no banned phrases,
     EVERY number must appear in the JSON record (one wrong number and the output is rejected)
  5. Put the personal data back on the server
  6. If any step fails: use the template text built from the JSON
"""
import concurrent.futures as cf
import json
import re

TIMEOUT_S = 8
BANNED = ("guaranteed delivery", "compensation", "definitely deliver", "guarantee")
PII_TAGS = {"customer_name": "[CUSTOMER_NAME]", "customer_address": "[ADDRESS]",
            "customer_phone": "[PHONE]", "customer_email": "[EMAIL]"}

SYSTEM = """You are a delivery operations assistant. Use only the data inside <data>.
1. Do not calculate, round or add any number that is not in <data>. Write numbers exactly as they appear in <data>.
2. Do not promise a delivery date other than promised_delivery_date.
3. Do not give reasons other than those in the reasons list.
4. If data is missing, answer: "Not enough data to explain this order."
5. Keep the placeholders [CUSTOMER_NAME], [ADDRESS], [PHONE], [EMAIL] unchanged.
6. explanation is written for staff: why the order is at risk and what to do (follow actions).
   draft_message is written for the customer: polite, announces the new delivery date in advance, never mentions probabilities or technical reasons.
Never use the phrases "guaranteed delivery" or "compensation". Do not use long dashes.
Return JSON {"explanation": "...", "draft_message": "..."}, each field at most 3 sentences."""

_POOL = cf.ThreadPoolExecutor(max_workers=2)
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


# ------------------------------------------------------------------ steps 2 and 5: mask and restore personal data
def mask(payload):
    """Returns (masked record, mapping placeholder -> real value). The mapping only lives in memory."""
    masked, mapping = dict(payload), {}
    for key, tag in PII_TAGS.items():
        if masked.get(key):
            mapping[tag] = str(masked[key])
            masked[key] = tag
    return masked, mapping


def unmask(text, mapping):
    for tag, value in mapping.items():
        text = text.replace(tag, value)
    return text


# ------------------------------------------------------------------ step 4: check the output
def _norm(tok):
    """'1,234' -> '1234', '76.6' -> '76.6', '2.0' -> '2' so that numbers compare equal however they are written."""
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", tok):
        tok = tok.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", tok):
        tok = tok.replace(".", "")
    tok = tok.replace(",", ".")
    try:
        return f"{float(tok):g}"
    except ValueError:
        return tok


def allowed_numbers(payload):
    return {_norm(t) for t in _NUM.findall(json.dumps(payload, ensure_ascii=False))}


def check_output(out, allowed):
    """List of problems; empty means the output is usable."""
    problems = []
    if not isinstance(out, dict) or not all(isinstance(out.get(k), str) and out[k].strip()
                                            for k in ("explanation", "draft_message")):
        return ["invalid JSON shape"]
    for key in ("explanation", "draft_message"):
        text = out[key]
        if len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]) > 3:
            problems.append(f"{key} has more than 3 sentences")
        low = text.lower()
        problems += [f"contains banned phrase '{b}'" for b in BANNED if b in low]
        extra = sorted({t for t in _NUM.findall(text) if _norm(t) not in allowed})
        if extra:
            problems.append(f"{key} has numbers that are not in the data: {', '.join(extra)}")
    return problems


# ------------------------------------------------------------------ step 3: call the LLM
def call_llm(system, prompt, json_mode=True):
    """Calls the LLM through the providers configured for the chatbot (Gemini, Groq). Over 8 seconds: retry once."""
    from views.chatbot import _generate          # shares API keys and provider fallback with the chatbot
    last = None
    for _ in range(2):
        try:
            return _POOL.submit(_generate, system, prompt, json_mode).result(timeout=TIMEOUT_S)
        except cf.TimeoutError:
            last = TimeoutError(f"the LLM did not answer within {TIMEOUT_S} seconds")
        except Exception as e:      # no API key, overloaded...: do not retry
            raise e
    raise last


def compose(payload, template):
    """Runs all 6 steps. template(payload) returns the fallback text {"explanation", "draft_message"}."""
    masked, mapping = mask(payload)
    try:
        text = call_llm(SYSTEM, "<data>\n" + json.dumps(masked, ensure_ascii=False, indent=1) + "\n</data>")
        m = re.search(r"\{.*\}", text or "", re.S)
        out = json.loads(m.group(0)) if m else None
        problems = check_output(out, allowed_numbers(masked))
        if problems:
            raise ValueError("; ".join(problems))
        result = {k: unmask(out[k].replace("—", "-").replace("–", "-"), mapping) for k in ("explanation", "draft_message")}
        return {**result, "source": "LLM", "note": "Passed the number and banned-phrase checks"}
    except Exception as e:
        result = {k: unmask(v, mapping) for k, v in template(masked).items()}
        return {**result, "source": "Template", "note": f"Template used because {str(e)[:160]}"}
