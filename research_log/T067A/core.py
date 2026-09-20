"""The one authorized first-safe selector; no fallback or offsets."""
def first_safe(probabilities,base):
    selected=next((k for k in range(base+1) if probabilities[k]>=.5),None)
    assert selected is not None,'No frozen first_safe; stop BLOCKED'
    return selected
