"""
tests/fixtures.py — shared test data and fakes (no network, no API keys).

  • SAMPLE_SEGMENTS: a synthetic 116-second Hinglish micro-drama episode with a
    strong reveal at 01:46–01:50 followed by an anticlimactic last line — so the
    "best cut" logic has something real to find.
  • RuleBasedFakeLLM: a LangChain chat model that answers each HookLens prompt
    with plausible JSON (dispatching on the TASK line of the system prompt).
  • FakeEmbeddings: deterministic hashed bag-of-words vectors, so ChromaDB
    tests run without downloading a sentence-transformers model.
"""
from __future__ import annotations

import hashlib
import json
import re

import tests._runner  # noqa: F401  (puts the project root on sys.path)

SAMPLE_DURATION = 116.0
_RAW = [
    (1.2, 4.0, "Arjun, ye locker ki chaabi tumhare coat mein kya kar rahi hai?", "Arjun, what is this locker key doing in your coat?"),
    (4.4, 7.8, "Meera, tum meri cheezon ki talaashi le rahi ho?", "Meera, are you searching through my things?"),
    (8.2, 11.5, "Jawab do! Ye wahi locker hai jo Papa ke marne ke baad gayab hua tha.", "Answer me! This is the same locker that disappeared after Papa died."),
    (12.0, 15.2, "Tum thak gayi ho. Chalo, so jao.", "You're tired. Come, go to sleep."),
    (19.5, 22.0, "Nahi Arjun, aaj nahi.", "No Arjun, not today."),
    (22.5, 26.0, "Maa ne kaha tha is ghar mein ek raaz hai.", "Mom said there is a secret in this house."),
    (26.5, 30.0, "Aur tum us raaz ka hissa ho.", "And you are part of that secret."),
    (31.0, 34.0, "Bas karo Meera, tum paagal ho gayi ho.", "Stop it Meera, you have gone mad."),
    (36.0, 38.5, "Theek hai. Main khud dekh loongi.", "Fine. I'll see for myself."),
    (45.0, 47.0, "Bank khulne mein abhi time hai.", "There's still time before the bank opens."),
    (47.5, 50.0, "Haan madam, aapka naam?", "Yes madam, your name?"),
    (50.5, 52.0, "Meera Malhotra.", "Meera Malhotra."),
    (52.5, 56.0, "Ye locker toh Arjun Malhotra ke naam pe hai.", "This locker is in Arjun Malhotra's name."),
    (56.5, 59.0, "Main unki biwi hoon. Kholiye.", "I'm his wife. Open it."),
    (66.0, 68.5, "Ye... ye toh Papa ki will hai.", "This... this is Papa's will."),
    (69.0, 73.0, "Saari property... Arjun ke naam? Papa ne kabhi aisa nahi kiya hota.", "All the property... in Arjun's name? Papa would never have done that."),
    (73.5, 77.0, "Signature bhi Papa ka nahi hai.", "The signature isn't even Papa's."),
    (78.0, 81.0, "Hello, Inspector Rathore? Mujhe ek case report karna hai.", "Hello, Inspector Rathore? I need to report a case."),
    (81.5, 85.0, "Meera, phone rakh do.", "Meera, put the phone down."),
    (85.5, 88.0, "Arjun? Tum yahan kaise?", "Arjun? How are you here?"),
    (88.5, 93.0, "Tumhe sach jaanna tha na? Toh suno.", "You wanted to know the truth, right? Then listen."),
    (93.5, 98.0, "Tumhare Papa ki maut accident nahi thi.", "Your Papa's death was not an accident."),
    (98.5, 102.5, "Aur jisne unhe maara... woh is waqt tumhare saath khadi hai.", "And the one who killed him... is standing with you right now."),
    (103.0, 106.0, "Kya matlab? Yahan toh sirf Maa hai...", "What do you mean? Only Mom is here..."),
    (106.5, 110.0, "Haan, Meera. Main.", "Yes, Meera. Me."),
    (111.0, 114.0, "Chalo, ab ghar chalte hain. Kal baat karenge.", "Come, let's go home now. We'll talk tomorrow."),
]
SAMPLE_SEGMENTS = [
    {"id": i, "start": s, "end": e, "text": t, "text_en": en} for i, (s, e, t, en) in enumerate(_RAW)
]
SAMPLE_SILENCES = [[15.3, 19.4], [38.6, 44.9], [59.1, 65.9], [114.1, 116.0]]


def sample_segments() -> list[dict]:
    return [dict(seg) for seg in SAMPLE_SEGMENTS]


# ── Fake LLM ──────────────────────────────────────────────────────────────────
_HIGH = ("maut", "maara", "main.", "jawab", "raaz", "signature", "will hai", "phone rakh", "paagal")
_LOW = ("chalo", "theek hai", "haan madam", "bank khulne", "meera malhotra.", "so jao")


def _rate(text: str) -> tuple[int, str]:
    low = text.lower()
    if any(word in low for word in ("maut", "maara", "main.")):
        return 9, "shock"
    if "!" in text or "paagal" in low or "phone rakh" in low:
        return 8, "anger"
    if any(word in low for word in ("raaz", "locker", "signature", "will")):
        return 7, "suspense"
    if any(word in low for word in _LOW):
        return 2, "calm"
    return 5, "suspense"


def fake_response(system: str, human: str) -> str:
    """Return the JSON (or text) HookLens expects for each prompt."""
    if "TASK: Score the HOOK" in system:
        return json.dumps({
            "sub_scores": {"conflict_tension": 7, "curiosity_gap": 8, "emotional_intensity": 6, "stakes_clarity": 5},
            "first_meaningful_line_s": 1.2,
            "open_question": "Why does Arjun have the missing locker key?",
            "reason": "Opens on an accusation about a hidden key, but we don't yet know what is at stake.",
            "strengths": ["instant confrontation", "specific object (the key)"],
            "weaknesses": ["stakes unclear", "no emotional peak"],
        })
    if "TASK: Rewrite the OPENING" in system:
        return json.dumps({"rewrites": [
            {"line": "Papa ki maut ke din ye locker gayab hua tha... aur aaj iski chaabi tumhare coat mein hai, Arjun?",
             "english": "This locker vanished the day Papa died... and today its key is in your coat, Arjun?",
             "technique": "stakes up front", "why": "Ties the key to Papa's death in the first line."},
            {"line": "(chaabi zameen par girti hai) Ye kiski chaabi hai, Arjun? Sach bolo!",
             "english": "(a key falls to the floor) Whose key is this, Arjun? Tell the truth!",
             "technique": "cold open on action", "why": "Visual sting plus a direct demand."},
            {"line": "Agar ye chaabi khuli, toh tumhara sach bhi khulega.",
             "english": "If this key opens, your truth opens too.",
             "technique": "threat", "why": "Adds a threat that raises curiosity."},
        ]})
    if "TASK: Score the CLIFFHANGER" in system:
        return json.dumps({
            "sub_scores": {"unresolved_question": 8, "rising_stakes": 8, "reveal_shock": 9,
                           "emotional_peak_at_cut": 4, "would_pay_next": 7},
            "best_cut_s": 109.7,
            "cut_reason": "End on Maa's confession 'Main.' — the last line deflates the reveal.",
            "open_question": "Why did Maa kill Papa?",
            "reason": "Huge reveal, but the episode continues for a calm line after the peak.",
            "strengths": ["confession reveal"], "weaknesses": ["cut lands after the peak"],
        })
    if "TASK: Write exactly 3 alternative ENDINGS" in system:
        return json.dumps({"endings": [
            {"ending": "Haan, Meera. Main. (Maa ke haath mein wahi chaabi chamakti hai)",
             "english": "Yes, Meera. Me. (the same key glints in Mom's hand)", "technique": "visual sting",
             "why": "Cuts on the peak with a new question."},
            {"ending": "Main. Aur agla number... Arjun ka hai.", "english": "Me. And the next one... is Arjun.",
             "technique": "threat", "why": "Raises stakes beyond the reveal."},
            {"ending": "(Inspector ka phone aata hai) Madam, aapke Papa zinda hain.",
             "english": "(the inspector calls) Madam, your Papa is alive.", "technique": "reversal",
             "why": "A second twist demands the next episode."},
        ]})
    if "TASK: Rate EVERY transcript segment" in system:
        ratings = []
        for match in re.finditer(r"id=(\d+) \[[^\]]*\] (.*)", human):
            intensity, emotion = _rate(match.group(2))
            ratings.append({"id": int(match.group(1)), "intensity": intensity, "emotion": emotion})
        return json.dumps({"ratings": ratings})
    if "TASK: Write the editor's verdict" in system:
        count = int(re.search(r"exactly (\d+) items", system).group(1))
        return json.dumps({
            "logline": "A wife chasing a missing locker uncovers that her own mother killed her father.",
            "summary": "Strong reveal-driven episode with a solid hook. The ending runs past its peak — cut on the confession.",
            "fixes": [{"title": f"Fix {i + 1}", "action": f"Concrete action {i + 1}."} for i in range(count)],
        })
    if "You are HookLens Chat" in system:
        return "The emotional peak is at [01:46] when Maa says 'Haan, Meera. Main.'"
    return "{}"


def make_fake_llm():
    """Build the fake chat model class lazily (needs langchain-core)."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class RuleBasedFakeLLM(BaseChatModel):
        calls: int = 0

        @property
        def _llm_type(self) -> str:
            return "hooklens-rule-based-fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls += 1
            system = str(messages[0].content) if messages else ""
            human = str(messages[1].content) if len(messages) > 1 else ""
            text = fake_response(system, human)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    return RuleBasedFakeLLM()


def fake_llm_factory():
    model = make_fake_llm()

    def factory(temperature: float, json_mode: bool):
        return model.bind(response_format={"type": "json_object"}) if json_mode else model

    return factory


# ── Fake embeddings ───────────────────────────────────────────────────────────
class FakeEmbeddings:
    """Hashed bag-of-words → 64-d unit vectors (deterministic, offline)."""

    dim = 64

    def _vector(self, text: str) -> list[float]:
        import numpy as np

        vec = np.zeros(self.dim)
        for word in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = float(np.linalg.norm(vec)) or 1.0
        return [float(v) for v in vec / norm]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)
