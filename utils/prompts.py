PERSONA_TEMPLATES = {
    "doctor": "You are Dr. Guardian, an experienced medical professional. Answer with evidence-based details.",
    "lawyer": "You are Counsel AI, an expert attorney. Provide thorough legal reasoning.",
    "coder":  "You are CodeSmith, a senior software engineer. Explain code clearly with best practices."
}

def build_prompt(persona, instruction, input_text, cot=False, multi_turn=False):
    sys_prompt = PERSONA_TEMPLATES.get(persona, "")
    user_prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}"
    if cot:
        user_prompt += "\n### Chain-of-thought:"
    if multi_turn:
        # Simple 2-turn example
        user_prompt = f"<s>[SYSTEM] {sys_prompt} [/SYSTEM]\n[USER] {user_prompt} [/USER]"
    return f"{sys_prompt}\n{user_prompt}"