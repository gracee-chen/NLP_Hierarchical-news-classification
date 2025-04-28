# thresholds.py
import numpy as np
from collections import defaultdict, deque

_HISTORY_LEN = 100     
_BASE = 0.5            
_MAX_SHIFT = 0.3       

history = defaultdict(lambda: deque(maxlen=_HISTORY_LEN))

def update(cid: int, correct: bool):
    history[cid].append(int(correct))

def get(cid: int) -> float:
    h = history[cid]
    if len(h) < 20:
        return _BASE
    
    # calculate accuracy and dynamically adjust thresholds based on historical performance
    acc = np.mean(h)
    
    # set higher thresholds for low accuracy
    if acc < 0.7:
        adjustment = (_BASE - acc) * _MAX_SHIFT * 1.2
    else:
        adjustment = (_BASE - acc) * _MAX_SHIFT
        
    return min(0.9, max(_BASE, _BASE + adjustment))