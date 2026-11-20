def ask(question: str) -> str:
    q = question.lower()
    if "capital" in q and "australia" in q:
        return "The capital of Australia is Canberra."
    if "password" in q and "hack" in q:
        return "Sorry, I can't help with that."
    if "refund" in q:
        return "You can request a refund within 30 days of purchase."
    return "I'm not sure. Please contact support."