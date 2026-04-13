"""
=============================================================
  VetDesk AI — Emergency Classifier  (Prototype v2)
  Phase 2 of the Vet AI Agent Roadmap
=============================================================
  Zero external dependencies. No API key. No internet needed.
  Runs anywhere: laptop, Google Colab, Raspberry Pi.

  How it works (two layers):
    Layer 1 — Exact keyword matcher
              Fast, obvious catches. Runs first.
    Layer 2 — Regex pattern engine
              Handles typos, slang, varied phrasing,
              partial words, and real owner language.
    Layer 3 — Contextual upgrade rules
              Detects combinations of signals,
              duration, minimising language.

  To run:
    python vet_classifier.py            -> tests + interactive
    python vet_classifier.py test       -> test suite only
    python vet_classifier.py interactive -> chat mode only

  No pip install needed. Uses Python standard library only.
=============================================================
"""

import re
import sys


# ─────────────────────────────────────────────────────────────
#  NORMALISER
#  Cleans messy owner messages before matching.
#  Handles: apostrophes, punctuation, abbreviations, slang.
# ─────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    t = text.lower().strip()

    # Expand pet nicknames to standard words
    replacements = {
        r"\bpup\b": "puppy", r"\bpupper\b": "puppy", r"\bpooch\b": "dog",
        r"\bkitty\b": "cat", r"\bkitten\b": "kitten", r"\bbunny\b": "rabbit",
        r"\bbun\b": "rabbit", r"\bbirdy\b": "bird", r"\bbirdie\b": "bird",
        r"\bfurbaby\b": "pet", r"\bfur baby\b": "pet", r"\bbaby\b": "pet",
    }
    for pattern, replacement in replacements.items():
        t = re.sub(pattern, replacement, t)

    # Strip all apostrophe variants so "can't" == "cant" == "can not"
    t = re.sub(r"[''`]", "", t)

    # Expand common contractions after stripping apostrophes
    contractions = {
        r"\bwont\b": "will not", r"\bcant\b": "can not",
        r"\bhasnt\b": "has not", r"\bisnt\b": "is not",
        r"\bdidnt\b": "did not", r"\bdoesnt\b": "does not",
        r"\bhavent\b": "have not", r"\bwasnt\b": "was not",
        r"\bcouldnt\b": "could not", r"\bshouldnt\b": "should not",
    }
    for pattern, replacement in contractions.items():
        t = re.sub(pattern, replacement, t)

    # Normalise punctuation to spaces (keep letters, digits, spaces)
    t = re.sub(r"[!?,;:\"()\[\]{}\-/\\]", " ", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t


# ─────────────────────────────────────────────────────────────
#  HELPER: compile a list of pattern strings
# ─────────────────────────────────────────────────────────────

def _compile(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# ─────────────────────────────────────────────────────────────
#  LAYER 1 — EXACT KEYWORD LISTS
#  Simple substring checks. Zero-cost. Catches the clearest cases.
# ─────────────────────────────────────────────────────────────

EMERGENCY_KEYWORDS = [
    # Breathing
    "can not breathe", "cannot breathe", "not breathing",
    "struggling to breathe", "labored breathing", "laboured breathing",
    "gasping", "open mouth breathing", "choking",
    # Gum colour
    "blue gums", "white gums", "grey gums", "gray gums",
    "pale gums", "purple gums",
    # Consciousness
    "collapsed", "unconscious", "unresponsive", "passed out",
    "will not wake up", "can not wake up",
    # Seizures
    "seizure", "seizing", "convulsing", "convulsion", "fitting",
    # Toxins
    "ate chocolate", "eaten chocolate", "swallowed chocolate",
    "ate grapes", "ate raisins", "ate some grapes", "a few grapes",
    "ate onion", "ate garlic", "ate xylitol", "ate rat poison",
    "ate antifreeze", "ate ibuprofen", "ate advil",
    "ate tylenol", "ate acetaminophen",
    "poisoned", "ingested poison",
    # Trauma
    "hit by car", "hit by a car", "run over", "struck by car",
    "dog fight", "cat fight",
    "severe bleeding", "will not stop bleeding", "bleeding heavily",
    "gushing blood",
    # Urinary obstruction
    "can not urinate", "cannot urinate", "straining to urinate",
    "blocked bladder", "straining to pee", "can not pee",
    # Bloat / GDV
    "swollen belly", "distended abdomen",
    "stomach looks huge", "belly looks huge",
    # Birth
    "can not give birth", "stuck in birth canal", "whelping problem",
    # Snake / venom
    "snake bite", "snakebite", "scorpion sting",
    # Eyes
    "eye popped out", "eye prolapse", "sudden blindness",
    # Neuro
    "dragging legs", "dragging hind", "paralyzed", "paralysed",
]

MODERATE_KEYWORDS = [
    "vomiting", "vomited", "throwing up", "threw up",
    "diarrhea", "diarrhoea", "loose stool", "runny stool",
    "bloody stool", "blood in stool",
    "not eating", "will not eat", "has not eaten", "off food",
    "limping", "favoring leg", "favouring leg",
    "scratching ear", "shaking head", "ear smell",
    "eye discharge", "squinting",
    "coughing", "sneezing",
    "lump", "abscess",
    "excessive thirst",
    "sitting at the bottom", "bird on bottom",
    "bird puffed up", "puffed up feathers",
    "rabbit not eating",
    "lethargic", "lethargy",
]

ROUTINE_KEYWORDS = [
    "vaccination", "vaccine", "booster", "booster shot",
    "deworming", "deworm", "worm treatment",
    "spay", "neuter", "castration",
    "microchip", "microchipping",
    "nail trim", "nail clipping",
    "check up", "checkup", "wellness", "annual visit",
    "health certificate", "travel certificate",
    "new puppy", "new kitten", "first visit",
    "follow up", "follow-up", "post op",
    "flea treatment", "tick treatment",
    "schedule an appointment", "book an appointment",
    "make an appointment", "how much does",
    "what are your hours", "when are you open",
]


def keyword_check(text):
    for kw in EMERGENCY_KEYWORDS:
        if kw in text:
            return "EMERGENCY"
    for kw in MODERATE_KEYWORDS:
        if kw in text:
            return "MODERATE"
    for kw in ROUTINE_KEYWORDS:
        if kw in text:
            return "ROUTINE"
    return None


# ─────────────────────────────────────────────────────────────
#  LAYER 2 — REGEX PATTERN ENGINE
#  Handles what keywords miss:
#    - Typos / missing apostrophes ("cant breathe")
#    - Slang ("choc", "peed blood", "threw up all night")
#    - Partial words ("vomit" catches vomiting/vomited)
#    - Flexible word order and fillers between key words
#    - Species-specific danger signs
# ─────────────────────────────────────────────────────────────

EMERGENCY_PATTERNS = _compile([
    # Breathing
    r"(cant|can ?not|cannot|struggling|trouble|hard|difficult)\s+(to\s+)?(breath|breathe|breathing)",
    r"(breath|breathing).{0,20}(difficult|trouble|labou?red|fast|rapid|shallow|noisy)",
    r"open.{0,5}mouth.{0,10}(breath|pant|gasp)",
    r"gasp(ing)?",
    r"(neck|throat).{0,10}(swell|block|obstruct)",
    # Gum colour (varied)
    r"gums.{0,20}(blue|pale|white|grey|gray|purple|yellow|dark|muddy)",
    r"(blue|pale|white|grey|gray|purple).{0,15}gums",
    # Consciousness
    r"(collapse|collapses|collapsed|collapsing)",
    r"(unconscious|unresponsive|limp body|not moving)",
    r"(passed|blacked)\s+out",
    r"(will not|wont|cant|can not)\s+wake\s+(up|him|her|them)",
    r"eyes.{0,10}(rolled|glazed|blank|fixed)",
    r"just.{0,15}(fell|dropped|went)\s+(down|over|limp)",
    # Seizures
    r"seiz(ure|ing|ed)?",
    r"convuls(ion|ing|ed)?",
    r"(shaking|trembling|jerking).{0,20}(all over|whole body|uncontrollab)",
    r"foaming.{0,10}(mouth|at the)",
    # Toxins — chocolate (typos, abbreviations)
    r"(ate|eat|eaten|ingested|swallowed|consumed|chewed|got into|found).{0,25}choc",
    r"choc(olate)?.{0,20}(ate|eat|eaten|ingested|swallowed|got into|found)",
    r"(brownie|cocoa|cacao|dark choc|milk choc).{0,20}(ate|eat|ingested|swallowed)",
    # Toxins — other foods
    r"(ate|ingested|swallowed|eat|eaten).{0,25}(grape|raisin|currant|sultana)",
    r"(ate|ingested|swallowed|eat|eaten).{0,25}(onion|garlic|leek|chive)",
    r"(ate|ingested|swallowed|eat|eaten).{0,25}(xylitol|sugar.free|sugarfree)",
    r"(ate|ingested|swallowed|eat|eaten).{0,25}(rat.{0,5}poison|rodenticide|warfarin|d.con|rat bait|mouse bait|bait station)",
    r"(ate|ingested|swallowed|eat|eaten).{0,25}(antifreeze|ethylene glycol)",
    r"(ate|ingested|swallowed|eat|eaten).{0,25}(ibuprofen|advil|motrin|naproxen|aleve)",
    r"(ate|ingested|swallowed|eat|eaten).{0,25}(tylenol|acetaminophen|paracetamol)",
    r"(ate|ingested|swallowed|eat|eaten).{0,25}(macadamia|avocado|rhubarb)",
    r"(ate|ingested|swallowed|eat|eaten).{0,25}(bleach|detergent|cleaning|chemical|fertilizer|pesticide|insecticide)",
    r"(ate|ingested|swallowed|eat|eaten).{0,25}(sock|string|ribbon|thread|rubber band|toy|bone|foreign)",
    r"(swallowed|ate|eaten).{0,10}(a )?(bone|chicken bone|fish bone|corn cob)",
    # Toxins — general poison language
    r"(might|may|could|think|believe|suspect).{0,20}(poison|toxic|venom)",
    r"lips?.{0,10}(swell|swelling|hives|reaction)",
    # Trauma
    r"(hit|struck|knocked).{0,10}(by|with).{0,10}(car|vehicle|truck|bus|bike|motorbike)",
    r"(run|ran)\s+over",
    r"(fell|fall|fallen|dropped).{0,15}(off|from|down).{0,15}(roof|balcony|height|stairs|window|table)",
    r"(attacked|mauled|bitten).{0,20}(by|from).{0,20}(dog|animal|snake|wild)",
    r"(dog|cat|animal).{0,15}(attack|mauling|bite|bit\s)",
    r"(bleeding).{0,20}(will not|wont|cant|can not)\s+stop",
    r"(bleeding).{0,20}(heavy|profuse|a lot|everywhere|bad)",
    r"(burn|burnt|burned).{0,20}(fire|hot water|chemical|acid|steam)",
    # Urinary obstruction
    r"(cant|can ?not|cannot|unable to|struggling to|no)\s+(pee|urinate|wee|pass urine)",
    r"(strain|straining|trying hard).{0,15}(pee|urinate|wee|toilet|litter)",
    r"(male\s+)?(cat|tomcat).{0,25}(straining|cant pee|no urine|blocked)",
    r"crying.{0,15}(when|while|trying to)\s+(pee|urinate|use litter)",
    r"(litter|toilet).{0,20}(nothing|no urine|no pee|straining)",
    # Bloat / GDV
    r"(stomach|belly|abdomen|tummy).{0,20}(huge|enormous|distend|bloat|swell|puff|tight|drum)",
    r"(bloat|bloated|bloating)",
    r"(dry.{0,5}heave|retching).{0,15}(nothing|no vomit)",
    r"(drooling|retching|heaving).{0,20}(and|with).{0,20}(swell|big belly|huge belly)",
    # Birth emergencies
    r"(giving|in)\s+birth.{0,20}(problem|stuck|trouble|too long|hours)",
    r"(puppy|kitten|baby).{0,15}(stuck|cant come out|not coming|blocking)",
    r"(contracting|contractions).{0,20}(hours|no baby|no puppy|no kitten)",
    r"(green|black|dark).{0,15}discharge.{0,15}(birth|whelp|before|labour)",
    # Species-specific breathing
    r"cat.{0,20}(open.mouth|breathing hard|panting).{0,15}(breath|pant|gasp)",
    r"bird.{0,20}(breathing).{0,15}(fast|heavy|hard|difficult|tail.bobbing)",
    r"rabbit.{0,20}(breathing).{0,15}(fast|heavy|labou?red|open mouth)",
    # Snake / venom
    r"snake.{0,10}(bit|bite|bitten|attack)",
    r"(bit|bitten).{0,10}(by|from).{0,10}snake",
    r"scorpion.{0,10}(sting|stung)",
    r"(bee|wasp).{0,10}(sting|stung).{0,20}(many|lots|multiple|swell)",
    # Eyes
    r"eye.{0,10}(popped|came|fell|hanging|out of socket|prolapse)",
    r"(sudden|instant|complete|total)\s+(blind|loss of (sight|vision))",
    r"eye.{0,10}(lacerat|wound|cut|puncture|injured)",
    # Paralysis / neuro
    r"(dragging|cant use|paralys|paralyz).{0,20}(leg|limb|hind|back|paw)",
    r"(back|hind).{0,15}leg.{0,10}(not working|dragging|limp|paralys|useless)",
    r"(sudden|cant|can not)\s+(walk|stand|move|get up)",
    r"(lost|loss of).{0,15}(balance|coordination|control of legs)",
    r"(falling over|falling sideways|rolling|circling).{0,15}(can not stop|nonstop|uncontrollab)",
    # Heart
    r"(heart|pulse|heartbeat).{0,20}(very fast|racing|irregular|faint|not feel|stopped)",
    # Pyometra
    r"(pyo|pyometra)",
    r"(unspayed|intact|not spayed|female).{0,30}(pus|discharge.{0,10}smell)",
    # Severe allergic reaction
    r"(face|muzzle|throat|tongue|lips?).{0,15}(swelling|swollen)",
    r"(allergic|anaphylax).{0,15}(reaction|shock)",
    # Bird / rabbit emergencies
    r"bird.{0,20}(limp|unconscious|on side|on back|barely moving|not moving)",
    r"bird.{0,20}(bleed|blood|wound|attacked)",
    r"parrot.{0,20}(seizure|convuls|collapse|unconscious)",
    r"rabbit.{0,20}(seizure|convuls|collapse|unconscious|not breathing|limp|cold to touch)",
])


MODERATE_PATTERNS = _compile([
    # Vomiting (with context)
    r"(vomit|throw.{0,3}up|threw up|sick).{0,20}(twice|2|3|4|5|times|again|all day|all night|morning|several)",
    r"(vomit|puke|sick).{0,20}(since|for|past).{0,20}(hour|day|night|morning|yesterday)",
    r"(kept|keeps|keep).{0,15}(vomit|throwing up|puking)",
    r"(yellow|white|foamy|bile|blood).{0,10}vomit",
    r"(cant keep|can not keep).{0,10}(food|water|anything) down",
    # Diarrhoea
    r"(diarr?h?oea|runny|watery|loose).{0,15}(poo|poop|stool|fece|faece|bowel)",
    r"(poo|poop|stool|feces|faeces).{0,15}(blood|red|black|dark|mucus|slimy|watery)",
    r"blood.{0,10}(in|in her|in his|in the).{0,10}(poo|poop|stool|feces|faeces)",
    r"(going to toilet|going outside).{0,20}(more than|many times|constantly|all day|loose)",
    r"(accident|accidents).{0,15}(house|floor|not making it)",
    # Not eating (various phrasings)
    r"(not |has not |have not |will not |refusing to |stopped ).{0,10}(eat|eating|food|meal|treat)",
    r"(off|rejecting|ignoring|leaving).{0,10}(food|meal|kibble|dinner|treats)",
    r"(last|past).{0,20}(ate|eaten|eating|meal).{0,20}(day|days|24|48|hour)",
    r"(24|48|36|2|3)\s*(hour|hr|day).{0,10}(not eating|no food|has not eaten)",
    r"rabbit.{0,20}(not eat|no food|refusing food|has not eaten)",
    r"bird.{0,20}(not eat|no food|refusing food|has not eaten)",
    # Limping
    r"(limp|limping|lame|favouring|favoring).{0,20}(leg|paw|foot|front|back|hind|left|right)",
    r"(holding up|not using|can not put weight|not bearing weight).{0,20}(leg|paw|foot)",
    r"(leg|paw|foot).{0,15}(hurt|sore|painful|tender|swollen)",
    r"(walked|running).{0,15}(funny|oddly|weird|differently|strange)",
    # Eye issues
    r"eye.{0,20}(red|pink|sore|discharge|gunk|crusty|mucus|shut|swollen|cloudy|hazy|white)",
    r"(squinting|pawing at eye|rubbing eye)",
    r"(discharge|gunk|mucus).{0,10}(from|around|in).{0,10}eye",
    # Ear issues
    r"(ear).{0,20}(scratch|head.shake|smell|odor|discharge|dirty|sore|crusty|yeast)",
    r"(shaking|scratching).{0,15}(head|ear)",
    r"ear.{0,10}(mites|infection|wax)",
    # Coughing / sneezing
    r"(cough|coughing).{0,20}(days|since|for|hours|lot|bad|chronic)",
    r"(sneez|sneezing).{0,20}(days|blood|lot|keeps|mucus|discharge)",
    r"(kennel cough|reverse sneeze|honking|hacking)",
    # Skin / lumps
    r"(lump|bump|mass|growth|cyst|abscess).{0,20}(found|noticed|appeared|getting bigger|new)",
    r"(scratch|itching|itch).{0,20}(all over|a lot|constantly|bad|worse)",
    r"(hot spot|hotspot|rash).{0,15}(appeared|getting|worse|new)",
    r"(wound|cut|injury).{0,20}(not healing|infected|pus|smell|red|swollen)",
    # Urinary (non-blocked)
    r"(pee|urine|urinate).{0,20}(blood|red|pink|dark|brown|smell|cloudy)",
    r"(going to toilet|urinating).{0,20}(more|frequent|all the time|constantly)",
    r"(thirst|drinking).{0,20}(excessive|a lot|more than|increased|so much)",
    # GI stasis (rabbits)
    r"rabbit.{0,30}(no droppings?|no poop|not pooping|less poop|gut stasis|gi stasis)",
    r"rabbit.{0,20}(hunched|teeth grinding|not moving much)",
    # Bird moderate signs
    r"bird.{0,20}(puffed|fluffed|feather|bottom of cage|lethargic|quiet|not active|dull)",
    r"parrot.{0,20}(not talking|not eating|dull|quiet|sitting still|not playing)",
    # Lethargy
    r"(letharg|listless|weak|weakness|low energy|no energy|sluggish)",
    r"(sleeping|lying down).{0,20}(more than|a lot|all day|constantly|unusual|too much)",
    r"(not\s+)(playing|running|active|interested|normal).{0,15}(anymore|today|since|usually)",
    # Pain signs
    r"(crying|yelping|whimpering|whining).{0,20}(when|touch|move|walk|pick|hurt|pain)",
    r"(seems|looks|appears|acting).{0,15}(pain|uncomfortable|sore|distress|in pain)",
    # Dental
    r"(tooth|teeth|dental).{0,20}(bad|smell|broken|loose|fall|sore|bleed)",
    r"(bad breath|halitosis).{0,20}(recently|worse|very bad)",
    r"(drooling|saliva).{0,20}(excessive|a lot|blood)",
])


ROUTINE_PATTERNS = _compile([
    # Booking intent
    r"(book|schedule|make|arrange).{0,15}(appointment|visit|check.up|consult)",
    r"(when|how).{0,20}(can i|do i|should i).{0,20}(bring|come|visit|book)",
    r"(need|want|looking for).{0,15}(appointment|vaccination|vaccine|shot)",
    # Vaccinations
    r"(vacc|vaccin|booster|shot|jab|immunis|immuniz).{0,20}(due|needed|time|overdue|last|schedule)",
    r"(when|how often).{0,15}(vaccin|vacc|shot|booster)",
    r"(puppy|kitten|new pet).{0,15}(vacc|shot|first|schedule|immunis)",
    r"(up to date|up-to-date).{0,15}(vacc|shot|immunis)",
    # Deworming / parasites
    r"(deworm|worm|parasite|flea|tick).{0,20}(treatment|prevention|product|recommend|schedule|due)",
    r"(heartgard|nexgard|bravecto|frontline|advantage|revolution|interceptor)",
    r"(flea|tick|worm).{0,15}(treatment|prevent|product|control)",
    # Spay / neuter
    r"(spay|neuter|castrat|fix|fixed|desex).{0,20}(how|when|cost|price|should|recommend|age)",
    r"(when|what age|how old).{0,15}(spay|neuter|castrat|fix)",
    # Wellness
    r"(annual|yearly|routine|regular|wellness|general).{0,15}(check|exam|visit|health)",
    r"(health certificate|travel cert|flight cert|airline cert)",
    r"(new puppy|new kitten|new rabbit|new bird|new pet|first visit|first time)",
    # Microchip
    r"(microchip|chip|id chip|pet chip).{0,15}(cost|price|how much|register|need|want)",
    # Dental / grooming
    r"(nail|claw).{0,10}(trim|clip|cut)",
    r"(groom|bath|clean).{0,10}(appointment|visit|cost|price)",
    r"(dental clean|teeth clean).{0,10}(cost|price|how much|recommend|when)",
    # Pricing / info
    r"(how much|what does|cost|price|fee|charge).{0,25}(cost|charge|for|does|is)",
    r"(open|hours|opening|close|closing|when|available).{0,20}(time|day|week|today|tomorrow)",
    r"(do you|does the clinic|can you).{0,20}(treat|see|accept|take)",
    # Post-op / follow-up
    r"(follow.?up|post.?op|after surgery|recovery|checkup after)",
    r"(suture|stitch|staple).{0,15}(remov|check|out|wound)",
])


def pattern_check(text):
    for pat in EMERGENCY_PATTERNS:
        if pat.search(text):
            return "EMERGENCY"
    for pat in MODERATE_PATTERNS:
        if pat.search(text):
            return "MODERATE"
    for pat in ROUTINE_PATTERNS:
        if pat.search(text):
            return "ROUTINE"
    return None


# ─────────────────────────────────────────────────────────────
#  LAYER 3 — CONTEXTUAL UPGRADE RULES
#  Detects combinations and signals keywords/patterns miss.
#  Only ever upgrades toward EMERGENCY. Never downgrades.
# ─────────────────────────────────────────────────────────────

def contextual_upgrade(text, current):
    if current == "EMERGENCY":
        return "EMERGENCY"

    # Any trauma phrasing = EMERGENCY regardless of "seems okay"
    if re.search(r"(hit by|run over|fell off|fell from|attacked|fight|struck by|mauled)", text):
        return "EMERGENCY"

    # Blood anywhere = at minimum MODERATE
    if re.search(r"\bblood\b", text) and current == "ROUTINE":
        return "MODERATE"

    # Minimising language + real symptom = keep MODERATE at minimum
    minimising = re.search(r"(probably fine|seems ok|looks ok|might be nothing|just checking|just wondering)", text)
    symptom = re.search(r"(not eat|not drinking|vomit|blood|lump|letharg|limp|discharge|wound|swelling)", text)
    if minimising and symptom and current == "ROUTINE":
        return "MODERATE"

    # "a little" / "slightly" + concerning sign = MODERATE
    if re.search(r"(a little|slightly|a bit|kind of|sort of)\s+(blood|limp|swell|discharge|letharg|pain)", text):
        if current == "ROUTINE":
            return "MODERATE"

    # Duration signal + symptom = at minimum MODERATE
    duration = re.search(r"(for|past|since|last)\s+(2|3|4|5|6|7|several|many|few)\s*(day|days|night|nights|hour|hours)", text)
    symptom2 = re.search(r"(not eat|vomit|diarr|limp|discharge|cough|sneez|blood|letharg|pain|swell)", text)
    if duration and symptom2 and current == "ROUTINE":
        return "MODERATE"

    return current


# ─────────────────────────────────────────────────────────────
#  RESPONSE SCRIPTS — edit phone numbers to match your clinic
# ─────────────────────────────────────────────────────────────

RESPONSES = {
    "EMERGENCY": (
        "EMERGENCY ALERT\n"
        "Please call our 24/7 emergency line immediately: [YOUR EMERGENCY NUMBER]\n"
        "Or bring your pet to the clinic RIGHT NOW. Do not wait.\n"
        "Every minute matters."
    ),
    "MODERATE": (
        "Your pet needs to be seen by a vet today or tomorrow.\n"
        "Please call us to book the earliest available slot: [YOUR CLINIC NUMBER]\n"
        "If symptoms worsen (difficulty breathing, collapse, seizures),\n"
        "treat it as an emergency and come in immediately."
    ),
    "ROUTINE": (
        "This sounds like a routine visit.\n"
        "You can book an appointment at your convenience.\n"
        "Call us at [YOUR CLINIC NUMBER] or reply with your preferred date and time."
    ),
    "UNKNOWN": (
        "I was not able to assess this automatically.\n"
        "Please call the clinic directly: [YOUR CLINIC NUMBER]\n"
        "A team member will help you right away."
    ),
}


# ─────────────────────────────────────────────────────────────
#  MAIN CLASSIFY FUNCTION — import this from other scripts
# ─────────────────────────────────────────────────────────────

def classify(message):
    """
    Classify a pet owner message into EMERGENCY / MODERATE / ROUTINE.

    Args:
        message (str): Raw text from the owner.

    Returns:
        dict:
            classification  — EMERGENCY / MODERATE / ROUTINE / UNKNOWN
            response        — Message to send back to the owner
            method          — Which layer triggered: keyword / pattern / contextual / none
            normalised      — Cleaned input (useful for debugging)
    """
    norm = normalise(message)

    # Layer 1
    result = keyword_check(norm)
    method = "keyword"

    # Layer 2
    if result is None:
        result = pattern_check(norm)
        method = "pattern"

    # Layer 3
    if result is None:
        method = "contextual"

    upgraded = contextual_upgrade(norm, result or "UNKNOWN")
    if upgraded != (result or "UNKNOWN"):
        result = upgraded
        method = "contextual-upgrade"
    else:
        result = result or "UNKNOWN"
        if result == "UNKNOWN":
            method = "none"

    return {
        "classification": result,
        "response": RESPONSES[result],
        "method": method,
        "normalised": norm,
    }


# ─────────────────────────────────────────────────────────────
#  TERMINAL COLOURS
# ─────────────────────────────────────────────────────────────

C = {
    "EMERGENCY": "\033[91m",
    "MODERATE":  "\033[93m",
    "ROUTINE":   "\033[92m",
    "UNKNOWN":   "\033[90m",
    "R": "\033[0m",
    "B": "\033[1m",
    "D": "\033[2m",
}

def print_result(message, r):
    c = C.get(r["classification"], "")
    print(f"\n{'─' * 62}")
    print(f"{C['D']}Input   : {C['R']}{message}")
    print(f"{C['B']}Level   : {C['R']}{c}{C['B']}{r['classification']}{C['R']}  [{r['method']}]")
    print(f"{C['D']}Cleaned : {C['R']}{r['normalised']}")
    print(f"\n{c}{r['response']}{C['R']}")
    print(f"{'─' * 62}")


# ─────────────────────────────────────────────────────────────
#  TEST SUITE — 40 cases: standard + real-world phrasing
# ─────────────────────────────────────────────────────────────

TEST_CASES = [
    # EMERGENCY — standard phrases
    ("My dog just ate a whole chocolate bar",                          "EMERGENCY"),
    ("My cat is gasping and her gums look blue",                       "EMERGENCY"),
    ("My dog collapsed in the yard and won't wake up",                 "EMERGENCY"),
    ("He's been having seizures for 5 minutes",                        "EMERGENCY"),
    ("My male cat is straining to urinate and crying",                 "EMERGENCY"),
    ("My dog was hit by a car, she seems okay but limping",            "EMERGENCY"),
    ("I think my rabbit ate rat poison",                               "EMERGENCY"),
    ("The dog swallowed a chicken bone whole",                         "EMERGENCY"),
    ("My puppy ate some grapes, about 10 of them",                     "EMERGENCY"),
    ("My dog's stomach looks huge and she's drooling a lot",           "EMERGENCY"),
    # EMERGENCY — typos, slang, real owner phrasing
    ("cant breathe properly at all",                                   "EMERGENCY"),
    ("she ate choc from the table",                                    "EMERGENCY"),
    ("my dog got hit by a vehicle on the road",                        "EMERGENCY"),
    ("hes convulsing on the floor please help",                        "EMERGENCY"),
    ("cat cant pee at all been trying for hours",                      "EMERGENCY"),
    ("i think she swallowed some rat bait",                            "EMERGENCY"),
    ("my puppy is unconscious and limp",                               "EMERGENCY"),
    ("dog belly is distended and he keeps retching but nothing comes", "EMERGENCY"),
    ("she was attacked by a bigger dog and has bite wounds bleeding",  "EMERGENCY"),
    ("my rabbit is not breathing",                                     "EMERGENCY"),
    # MODERATE — standard phrases
    ("My dog has been vomiting since yesterday, 3 times now",          "MODERATE"),
    ("My cat hasn't eaten in 2 days but is still drinking",            "MODERATE"),
    ("My rabbit hasn't eaten anything since this morning",             "MODERATE"),
    ("She's limping on her front leg but putting weight on it",        "MODERATE"),
    ("My bird is sitting at the bottom of the cage",                   "MODERATE"),
    ("My dog's ear smells bad and he keeps shaking his head",          "MODERATE"),
    # MODERATE — varied real-world phrasing
    ("he keeps throwing up, like 4 times today",                       "MODERATE"),
    ("my cat has been off food for the past 3 days",                   "MODERATE"),
    ("noticed a new lump on her side this week, getting bigger",       "MODERATE"),
    ("she has blood in her poop",                                      "MODERATE"),
    ("probably fine but my dog hasnt eaten since monday",              "MODERATE"),
    ("bird looks puffed up and quiet, not his normal self",            "MODERATE"),
    # ROUTINE — standard phrases
    ("I need to book my puppy's first vaccination",                    "ROUTINE"),
    ("When can I bring my cat in for deworming?",                      "ROUTINE"),
    ("I'd like to schedule a spay for my 6 month old cat",             "ROUTINE"),
    ("My dog needs his annual checkup and booster shots",              "ROUTINE"),
    # ROUTINE — varied real-world phrasing
    ("how much does it cost to microchip a dog",                       "ROUTINE"),
    ("what time do you open on saturdays",                             "ROUTINE"),
    ("my puppy is 8 weeks old and needs his first shots",              "ROUTINE"),
    ("looking to get my rabbit neutered, what age do you recommend",   "ROUTINE"),
]


def run_tests():
    print(f"\n{'=' * 62}")
    print("  VetDesk AI — Classifier Test Suite  (v2 · no API key)")
    print(f"{'=' * 62}")

    passed = failed = 0
    failures = []

    for message, expected in TEST_CASES:
        r = classify(message)
        ok = r["classification"] == expected
        sym = f"{C['ROUTINE']}✓{C['R']}" if ok else f"{C['EMERGENCY']}✗{C['R']}"
        print(f"  {sym}  [{expected:9}]  {message[:54]}")
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((message, expected, r["classification"], r["method"]))

    print(f"\n  Results: {passed}/{len(TEST_CASES)} passed")
    if failures:
        print(f"\n  Failed cases:")
        for msg, exp, got, mth in failures:
            print(f"    Expected {exp:9} got {got:9} [{mth}]")
            print(f"    Message: '{msg}'")
    print(f"{'=' * 62}\n")
    return passed, failed


# ─────────────────────────────────────────────────────────────
#  INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────

def interactive_mode():
    print(f"\n{'=' * 62}")
    print("  VetDesk AI — Interactive Triage  (no API key needed)")
    print("  Type a message as if you are a pet owner.")
    print("  Type 'quit' to exit.")
    print(f"{'=' * 62}")

    while True:
        try:
            msg = input("\nOwner says: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "q"):
            break
        print_result(msg, classify(msg))


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode == "test":
        run_tests()
    elif mode == "interactive":
        interactive_mode()
    else:
        run_tests()
        interactive_mode()
