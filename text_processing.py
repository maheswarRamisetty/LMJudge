import re
from jcjs_coherence.utils.utils import simple_normalize

def minimal_clean(text):
    t = simple_normalize(text)
    t = re.sub(r'\s+', ' ', t)
    return t