from typing import List, Dict
import math

def apply_simple_fairness(candidates: List[Dict], protected_field: str, max_share: float = 0.5):
    """
    Simple post-hoc re-ranking to prevent > max_share of top-K coming from same protected attribute value.
    E.g., protected_field = 'university' would limit a single university appearing more than max_share.
    """
    counts={}
    output=[]
    for c in candidates:
        val = c.get("metadata",{}).get(protected_field) or "unknown"
        counts.setdefault(val, 0)
        # allow if not exceeding share of final list
        if counts[val] / (len(output)+1) > max_share:
            # demote by reducing score
            c["final_score"] *= 0.8
        else:
            counts[val]+=1
        output.append(c)
    # resort
    return sorted(output, key=lambda x: x["final_score"], reverse=True)
