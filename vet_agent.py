"""
=============================================================
  VetDesk AI — Phase 3 Agent  (v2 — improved replies)
  vet_agent.py
=============================================================
  Combines:
    Phase 2  vet_classifier.py     → triage classification
    Phase 3  vet_sop_knowledge.py  → SOP knowledge base
             TF-IDF RAG engine     → question answering
             Intent router         → direct answers for common Qs
             Response formatter    → conversational replies

  What improved in v2:
    - Specific intent router intercepts hours / fasting /
      services / toxic food questions BEFORE TF-IDF runs,
      preventing "hours" matching fasting chunks etc.
    - Response formatter wraps raw chunks in warm,
      client-facing language instead of dumping SOP text.

  To run:
    pip install scikit-learn
    python vet_agent.py
=============================================================
"""

import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vet_classifier import classify as triage_classify
from vet_sop_knowledge import (
    CHUNKS, CLINIC_NAME, CLINIC_PHONE,
    EMERGENCY_LINE, WHATSAPP, ADDRESS
)


# ─────────────────────────────────────────────────────────────
#  TF-IDF RAG ENGINE
# ─────────────────────────────────────────────────────────────

class VetRAG:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(chunks)

    def search(self, query, top_k=2, threshold=0.08):
        query_vec = self.vectorizer.transform([query])
        scores    = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx   = scores.argsort()[::-1][:top_k]
        return [
            (round(float(scores[i]), 3), self.chunks[i])
            for i in top_idx
            if scores[i] >= threshold
        ]


# ─────────────────────────────────────────────────────────────
#  BOOKING STATE MACHINE
# ─────────────────────────────────────────────────────────────

BOOKING_FIELDS     = ["pet_name", "species", "owner_name", "phone", "reason", "date", "time"]
BOOKING_QUESTIONS  = {
    "pet_name":   "What is your pet's name?",
    "species":    "What type of animal is it? (dog, cat, rabbit, or bird)",
    "owner_name": "What is your name?",
    "phone":      "What is the best phone number to reach you?",
    "reason":     "What is the reason for the visit?",
    "date":       "What date would you prefer? (e.g. Monday April 14)",
    "time":       "What time works best? Morning (9–11 am) or afternoon (1–4 pm)?",
}

class BookingSession:
    def __init__(self):
        self.active        = False
        self.data          = {f: None for f in BOOKING_FIELDS}
        self.current_field = None

    def start(self):
        self.active        = True
        self.current_field = "pet_name"
        return (
            f"Of course! I'd be happy to book an appointment at {CLINIC_NAME}.\n\n"
            + BOOKING_QUESTIONS["pet_name"]
        )

    def next_field(self):
        for field in BOOKING_FIELDS:
            if self.data[field] is None:
                return field
        return None

    def receive(self, message):
        if self.current_field:
            self.data[self.current_field] = message.strip()
        nxt = self.next_field()
        if nxt:
            self.current_field = nxt
            return BOOKING_QUESTIONS[nxt]
        return self._confirm()

    def _confirm(self):
        self.active = False
        d = self.data
        return (
            f"Perfect — you're all booked! Here's your summary:\n\n"
            f"  🐾 Pet:     {d['pet_name']} ({d['species']})\n"
            f"  👤 Owner:   {d['owner_name']}\n"
            f"  📞 Phone:   {d['phone']}\n"
            f"  📋 Reason:  {d['reason']}\n"
            f"  📅 Date:    {d['date']}\n"
            f"  🕐 Time:    {d['time']}\n\n"
            f"Please arrive 10 minutes early and bring any previous vaccination records.\n"
            f"If anything changes, call us at {CLINIC_PHONE}. See you soon! 🐾"
        )

    def reset(self):
        self.__init__()


# ─────────────────────────────────────────────────────────────
#  INTENT DETECTION
# ─────────────────────────────────────────────────────────────

BOOKING_PATTERNS = [
    r"(book|schedule|make|arrange|set up).{0,20}(appointment|visit|consult)",
    r"(want|need|like).{0,15}(appointment|book|come in)",
    r"(can i|how do i|where do i).{0,15}(book|schedule)",
    r"(next|first|earliest).{0,10}(available|slot|appointment|opening)",
]

INFO_PATTERNS = [
    r"(how much|what does|cost|price|fee|charge|expensive)",
    r"(what time|when|hours|open|close|opening|closing)",
    r"(do you|can you|do you treat|do you accept|do you see)",
    r"(what is|what are|how do|how long|how often)",
    r"(tell me|explain|describe|what should|need to know)",
]

def detect_intent(message):
    norm = message.lower()
    for pat in BOOKING_PATTERNS:
        if re.search(pat, norm): return "booking"
    for pat in INFO_PATTERNS:
        if re.search(pat, norm): return "info"
    return "unclear"


# ─────────────────────────────────────────────────────────────
#  SPECIFIC INTENT ROUTER
#  Intercepts well-known question types before TF-IDF runs.
#  Returns a formatted reply string, or None to fall through.
#
#  Why this matters:
#    TF-IDF matches on word overlap. "What are your hours?"
#    contains "hours" which also appears in fasting chunks
#    ("8 to 12 hours before surgery"). This router catches
#    those questions first so TF-IDF never sees them.
# ─────────────────────────────────────────────────────────────

HOURS_PATTERNS = [
    r"(what|when).{0,10}(hour|open|close|opening|closing|time you)",
    r"(are you|do you).{0,10}open",
    r"(open|close).{0,10}(today|tomorrow|saturday|sunday|monday|weekday|weekend)",
    r"(what time|opening time|closing time|office hour)",
]

FASTING_PATTERNS = [
    r"(fast|fasting|no food|withhold food|before surgery|before operation|before procedure|before anaesth)",
    r"(eat|food).{0,15}(before surgery|before operation|before procedure)",
    r"(how long).{0,20}(fast|no food|not eat|not feeding)",
]

SERVICES_PATTERNS = [
    r"(what service|what do you offer|what can you|do you offer|what do you do|do you provide)",
    r"(do you treat|do you see|do you accept|do you handle)",
    r"(what animals|what pets|which animals|which pets)",
]

TOXIC_PATTERNS = [
    r"(what|which|is).{0,20}(toxic|poison|dangerous|harmful|bad).{0,20}(dog|cat|rabbit|bird|pet)",
    r"(toxic|poison|dangerous).{0,10}(food|plant|substance)",
    r"can (a |my )?(dog|cat|rabbit|bird) eat",
]

PRICE_PATTERNS = [
    r"(how much|what.{0,5}cost|price|fee|charge|expensive|affordable)",
]


def route_specific_intent(message: str):
    """Returns a formatted reply string or None."""
    norm = message.lower()

    # ── Opening hours ────────────────────────────────────────
    if any(re.search(p, norm) for p in HOURS_PATTERNS):
        if re.search(r"sun(day)?", norm):
            return (
                f"We're closed on Sundays, but our 24/7 emergency line is always "
                f"available at {EMERGENCY_LINE} for urgent cases.\n\n"
                f"Our regular hours are Monday–Friday 8:00 AM–6:00 PM and "
                f"Saturday 9:00 AM–4:00 PM.\n\n"
                f"Would you like to book an appointment for another day?"
            )
        if re.search(r"sat(urday)?", norm):
            return (
                f"Yes, we're open on Saturdays from 9:00 AM to 4:00 PM!\n\n"
                f"We're also open Monday to Friday, 8:00 AM to 6:00 PM.\n\n"
                f"Would you like to book an appointment?"
            )
        if re.search(r"weekend", norm):
            return (
                f"Our weekend hours:\n\n"
                f"• Saturday: 9:00 AM – 4:00 PM ✅\n"
                f"• Sunday: Closed\n\n"
                f"For Sunday emergencies, our 24/7 line is always available: {EMERGENCY_LINE}"
            )
        # General hours
        return (
            f"Our opening hours are:\n\n"
            f"• Monday – Friday: 8:00 AM – 6:00 PM\n"
            f"• Saturday: 9:00 AM – 4:00 PM\n"
            f"• Sunday: Closed\n\n"
            f"For life-threatening emergencies outside these hours, our "
            f"24/7 emergency line is always available: {EMERGENCY_LINE}\n\n"
            f"Would you like to book an appointment?"
        )

    # ── Pre-surgery fasting ──────────────────────────────────
    if any(re.search(p, norm) for p in FASTING_PATTERNS):
        is_rabbit = re.search(r"rabbit|bunny", norm)
        is_bird   = re.search(r"bird|parrot|budgie|canary|cockatiel", norm)
        is_cat    = re.search(r"\bcat\b|kitten|feline", norm)

        if is_rabbit:
            return (
                "Important: please do NOT fast your rabbit before surgery. "
                "Unlike dogs and cats, rabbits must keep eating right up until their procedure — "
                "withholding food can trigger dangerous GI stasis.\n\n"
                "Water should also stay available at all times.\n\n"
                "Our vet team will walk you through the full prep instructions when you book."
            )
        if is_bird:
            return (
                "Birds only need a short fast of 2–4 hours before surgery — "
                "much less than dogs or cats. Never withhold water from a bird.\n\n"
                "Our vet will confirm the exact timing for your bird's procedure."
            )
        if is_cat:
            return (
                f"For cats before surgery:\n\n"
                f"• No food for 8–12 hours before the procedure\n"
                f"• No water for 4 hours before\n\n"
                f"Our team will confirm the exact timing when you book. "
                f"Any questions? Call us at {CLINIC_PHONE}."
            )
        # Dog or general
        return (
            f"Pre-surgery fasting rules by species:\n\n"
            f"• Dogs & cats: no food 8–12 hours before, no water 4 hours before\n"
            f"• Rabbits: do NOT fast — keep food and water available right up to the procedure\n"
            f"• Birds: fast for 2–4 hours only, never withhold water\n\n"
            f"Our team will confirm the schedule for your pet when you book. "
            f"Call us at {CLINIC_PHONE} with any questions."
        )

    # ── Services overview ────────────────────────────────────
    if any(re.search(p, norm) for p in SERVICES_PATTERNS):
        return (
            f"Here's what we offer at {CLINIC_NAME}:\n\n"
            f"• Consultations & wellness exams\n"
            f"• Vaccinations & deworming\n"
            f"• Surgery (spay, neuter, wound repair, mass removal)\n"
            f"• Dental cleaning under anaesthesia\n"
            f"• Diagnostics — blood tests, X-rays, urinalysis\n"
            f"• Microchipping & travel health certificates\n"
            f"• Hospitalisation & post-surgical monitoring\n"
            f"• Nail trims & minor grooming\n\n"
            f"We treat dogs, cats, rabbits, and pet birds.\n\n"
            f"Is there a specific service you'd like to book or learn more about?"
        )

    # ── Toxic foods info ─────────────────────────────────────
    if any(re.search(p, norm) for p in TOXIC_PATTERNS):
        return (
            f"Common household items toxic to pets:\n\n"
            f"🐕 Dogs & cats: chocolate, grapes & raisins, onions, garlic, "
            f"xylitol (in sugar-free gum/candy), macadamia nuts, avocado, "
            f"ibuprofen (Advil/Motrin), acetaminophen (Tylenol), antifreeze, rat poison\n\n"
            f"🐇 Rabbits: onions, avocado, iceberg lettuce, chocolate, "
            f"rhubarb, potatoes\n\n"
            f"🐦 Birds: avocado, chocolate, onions, caffeine, alcohol, "
            f"xylitol, apple seeds\n\n"
            f"⚠️ If your pet has eaten any of these, call our emergency line "
            f"immediately: {EMERGENCY_LINE}"
        )

    return None  # No specific match — fall through to TF-IDF


# ─────────────────────────────────────────────────────────────
#  RESPONSE FORMATTER
#  Wraps raw retrieved SOP chunks in conversational language.
#  Called only when the intent router found no specific match.
# ─────────────────────────────────────────────────────────────

def format_rag_response(results: list, query: str, triage_level: str, rag=None) -> str:
    """
    Takes TF-IDF results and formats a warm client-facing reply.

    Args:
        results:       list of (score, chunk_text) from RAG.search()
        query:         the owner's original message
        triage_level:  EMERGENCY / MODERATE / ROUTINE / UNKNOWN

    Returns:
        Formatted reply string.
    """
    if not results:
        return None

    top_score, top_chunk = results[0]
    query_lower = query.lower()
    is_price_q  = any(re.search(p, query_lower) for p in PRICE_PATTERNS)
    has_price   = bool(re.search(r"\$[\d,]+ ?(–|-|to) ?\$[\d,]+", top_chunk))

    # ── Pricing reply ────────────────────────────────────────
    # If asking price AND about a specific service, search for price chunk
    if is_price_q:
        # Re-run search with price-biased query to get cost chunks to the top
        if rag:
            price_query = query + " cost price dollars"
            price_results = rag.search(price_query, top_k=3, threshold=0.08)
            # Use price results if they actually contain dollar amounts
            price_hits = [(s, c) for s, c in price_results if re.search(r"\$[\d,]+", c)]
            if price_hits:
                results = price_hits

        top_score, top_chunk = results[0]
        has_price = bool(re.search(r"\$[\d,]+ ?(–|-|to) ?\$[\d,]+", top_chunk))

    if is_price_q and has_price:
        price_match = re.search(r"\$[\d,]+ ?(–|-|to) ?\$[\d,]+[^.]*", top_chunk)
        price_str   = price_match.group(0).strip() if price_match else "available on request"
        service     = re.split(r" costs? | is priced", top_chunk, maxsplit=1)[0].strip()

        reply = (
            f"Great question! {service} typically runs {price_str}.\n\n"
            f"That's an estimate — the exact cost depends on your pet's "
            f"size and any additional care needed on the day. "
            f"We'll always confirm pricing before any procedure.\n\n"
        )
        # Add a second relevant price chunk if available
        if len(results) > 1 and results[1][0] > 0.12:
            second_chunk = results[1][1]
            if re.search(r"\$[\d,]+", second_chunk) and second_chunk != top_chunk:
                reply += f"Also worth knowing: {second_chunk}\n\n"

        reply += f"Would you like to book? Say 'book appointment' or call {CLINIC_PHONE}."
        return reply

    # ── Vaccination schedule reply ───────────────────────────
    if re.search(r"(vacc|booster|shot|jab|immunis)", query_lower):
        # If asking about puppies/kittens, search specifically for the puppy chunk
        if re.search(r"puppy|puppies|kitten|young|baby|new", query_lower):
            if rag:
                puppy_results = rag.search("puppy vaccination schedule first shot weeks", top_k=2, threshold=0.0)
                results = puppy_results if puppy_results else results
        reply = f"Here's the vaccination info for your pet:\n\n{results[0][1]}"
        if len(results) > 1 and results[1][0] > 0.12:
            reply += f"\n\n{results[1][1]}"
        reply += f"\n\nWould you like to book a vaccination visit? Call {CLINIC_PHONE} or say 'book appointment'."
        return reply

    # ── Animals / species queries ────────────────────────────
    if re.search(r"(do you treat|do you see|can you treat|accept|take care of)", query_lower):
        reply = top_chunk
        if "we do not" not in top_chunk.lower():
            reply += f"\n\nWould you like to book an appointment? Just say 'book appointment'."
        return reply

    # ── General info — conversational wrap ──────────────────
    # Soften clinical/staff-facing phrasing
    cleaned = re.sub(r"^(The clinic |We )", "", top_chunk, flags=re.IGNORECASE)
    cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else top_chunk

    reply = cleaned

    # Add second chunk only if meaningfully different and relevant
    if len(results) > 1 and results[1][0] > 0.15:
        second = results[1][1]
        # Don't add if both chunks are just price lines
        both_prices = re.search(r"\$", second) and re.search(r"\$", top_chunk)
        if not both_prices and second != top_chunk:
            reply += f"\n\n{second}"

    # Triage advisory
    if triage_level == "MODERATE":
        reply += (
            f"\n\nBased on what you've described, your pet should be seen "
            f"today or tomorrow. Please call {CLINIC_PHONE} to book the "
            f"earliest available slot."
        )
    else:
        reply += f"\n\nAnything else I can help with? You can also call us directly at {CLINIC_PHONE}."

    return reply


# ─────────────────────────────────────────────────────────────
#  MAIN AGENT
# ─────────────────────────────────────────────────────────────

class VetAgent:
    """
    The complete VetDesk AI agent.
    Instantiate once, call respond(message) for each message.
    """

    def __init__(self):
        print("Building knowledge index...", end=" ", flush=True)
        self.rag     = VetRAG(CHUNKS)
        self.booking = BookingSession()
        print("ready.")

    def respond(self, message: str) -> str:
        """
        Decision flow per message:
          1. Booking in progress          → continue collecting fields
          2. Triage classifier fires EMERGENCY → urgent message
          3. Specific intent router       → direct formatted reply
          4. Booking intent detected      → start booking flow
          5. TF-IDF RAG search            → formatted reply from SOP
          6. No match                     → fallback with phone number
        """

        # ── 1. Booking in progress ───────────────────────────
        if self.booking.active:
            return self.booking.receive(message)

        # ── 2. Triage check ──────────────────────────────────
        triage = triage_classify(message)
        level  = triage["classification"]

        # Safety: if question is clearly informational (asking ABOUT toxins,
        # not reporting an ingestion), don't fire the emergency response.
        # e.g. "is chocolate dangerous?" vs "my dog ate chocolate"
        is_info_about_toxins = bool(re.search(
            r"^(is|are|what|which|can|does|do).{0,30}(toxic|poison|dangerous|harmful|bad|safe)",
            message.lower()
        ))
        if level == "EMERGENCY" and is_info_about_toxins:
            level = "ROUTINE"   # treat as an info query, router will handle it

        if level == "EMERGENCY":
            return (
                f"🚨 This sounds like an emergency — please act immediately.\n\n"
                f"Call our 24/7 emergency line right now: {EMERGENCY_LINE}\n"
                f"Or bring your pet to us at {ADDRESS} as fast as you can.\n\n"
                f"Do not wait for an appointment. Every minute counts."
            )

        # ── 3. Specific intent router ────────────────────────
        routed = route_specific_intent(message)
        if routed:
            return routed

        # ── 4. Booking intent ────────────────────────────────
        if detect_intent(message) == "booking":
            return self.booking.start()

        # ── 5. TF-IDF RAG search ─────────────────────────────
        results = self.rag.search(message, top_k=2, threshold=0.08)
        if results:
            formatted = format_rag_response(results, message, level, rag=self.rag)
            if formatted:
                return formatted

        # ── 6. Fallback ──────────────────────────────────────
        fallback = (
            f"I want to make sure you get the right answer for that.\n"
            f"Please call us at {CLINIC_PHONE} and one of our team "
            f"members will help you right away."
        )
        if level == "MODERATE":
            fallback += (
                f"\n\nBased on what you've described, please try to get "
                f"your pet seen today or tomorrow."
            )
        return fallback


# ─────────────────────────────────────────────────────────────
#  TERMINAL COLOURS
# ─────────────────────────────────────────────────────────────

C = {
    "agent":  "\033[96m",
    "owner":  "\033[97m",
    "system": "\033[90m",
    "R":      "\033[0m",
    "B":      "\033[1m",
}

def print_exchange(role, text):
    label     = f"{C['agent']}{C['B']}VetDesk AI :{C['R']} " if role == "agent" else f"{C['owner']}You        :{C['R']} "
    indented  = text.replace("\n", "\n             ")
    print(f"\n{label}{indented}")


# ─────────────────────────────────────────────────────────────
#  TEST SUITE
# ─────────────────────────────────────────────────────────────

TEST_CASES = [
    # Emergency
    ("my cat is gasping and her gums are blue",     "emergency"),
    # Hours — the key fix
    ("what are your hours",                          "Monday"),
    ("are you open on saturday",                     "9:00 AM"),
    ("are you open on sundays",                      "closed"),
    # Fasting — the key fix
    ("how long do i fast my dog before surgery",     "8"),
    ("should i fast my rabbit before surgery",       "NOT"),
    ("fasting rules for cats",                       "8"),
    # Pricing
    ("how much does it cost to spay a cat",          "200"),
    ("what is the price of a rabies vaccine",        "15"),
    ("how much is microchipping",                    "25"),
    # Services
    ("what services do you offer",                   "Consultation"),
    ("do you treat rabbits",                         "rabbit"),
    ("do you see birds",                             "bird"),
    # Toxic foods
    ("is chocolate dangerous for dogs",              "chocolate"),
    ("what foods are toxic to cats",                 "toxic"),
    # Vaccination
    ("when does my puppy need vaccines",             "6"),
    # Booking
    ("id like to book an appointment",               "book"),
    # Rabbit not eating — MODERATE
    ("my rabbit hasnt eaten anything today",         "GI stasis"),
]

def run_tests(agent):
    print(f"\n{'=' * 65}")
    print("  VetDesk AI — Agent Test Suite  (v2)")
    print(f"{'=' * 65}")

    passed = failed = 0
    failures = []

    for message, expected_fragment in TEST_CASES:
        reply = agent.respond(message)
        agent.booking.reset()
        ok  = expected_fragment.lower() in reply.lower()
        sym = "\033[92m✓\033[0m" if ok else "\033[91m✗\033[0m"
        print(f"  {sym}  {message[:55]}")
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((message, expected_fragment, reply[:100]))

    print(f"\n  Results: {passed}/{len(TEST_CASES)} passed")
    if failures:
        print("\n  Failed cases:")
        for msg, exp, got in failures:
            print(f"    Query   : '{msg}'")
            print(f"    Expected: '{exp}' in reply")
            print(f"    Got     : '{got}...'")
    print(f"{'=' * 65}\n")
    return passed, failed


# ─────────────────────────────────────────────────────────────
#  INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────

def interactive_mode(agent):
    print(f"\n{'=' * 65}")
    print(f"  VetDesk AI — Interactive  (v2 · no API key)")
    print(f"  Type 'quit' to exit | 'reset' to clear booking")
    print(f"{'=' * 65}")

    while True:
        try:
            msg = input(f"\n{C['owner']}You: {C['R']}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not msg: continue
        if msg.lower() in ("quit", "exit", "q"): break
        if msg.lower() == "reset":
            agent.booking.reset()
            print(f"{C['system']}Booking cleared.{C['R']}")
            continue
        print_exchange("agent", agent.respond(msg))


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = VetAgent()
    mode  = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode == "test":         run_tests(agent)
    elif mode == "interactive": interactive_mode(agent)
    else:
        run_tests(agent)
        interactive_mode(agent)
