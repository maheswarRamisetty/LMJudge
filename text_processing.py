import re
from utils import simple_normalize

def minimal_clean(text):
    t = simple_normalize(text)
    t = re.sub(r'\s+', ' ', t)
    return t
