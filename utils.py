import re
import numpy as np

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def simple_normalize(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def to_pairs(flat):
    pairs = []
    i = 0
    while i < len(flat):
        idx = flat[i]
        vals = flat[i + 1] if i + 1 < len(flat) else []
        pairs.append((idx, vals))
        i += 2
    return pairs

def parse_json(json_str: str) -> str:
    if not isinstance(json_str, str):
        return ""

    json_str = json_str.strip()
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return data.get("reasoning", "")
        return ""
    except Exception:
        pass

    match = re.search(
        r'"reasoning"\s*:\s*"(.+?)"\s*,\s*"table_data"',
        json_str,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()

    match = re.search(
        r'"reasoning"\s*:\s*"(.+)',
        json_str,
        re.DOTALL
    )
    if match:
        return match.group(1).strip().rstrip('"').rstrip('}')

    # Give up safely
    return ""
