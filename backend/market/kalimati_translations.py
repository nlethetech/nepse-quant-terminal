# Nepali to English translations for Kalimati Market commodities
# Keys are matched as substrings of the full Nepali name (longest match wins)
from __future__ import annotations
import re

# ── Main commodity map (longer/more-specific entries first) ──────────────────
# Order matters: more specific entries should be checked before generic ones.
# We sort by length descending at runtime, so just keep them grouped logically.

VEGETABLE_MAP: dict[str, str] = {
    # ── Greens / Leafy ────────────────────────────────────────────────────────
    "पालुङ्गो":       "Spinach",
    "पालुंगो":        "Spinach",
    "पालक":           "Spinach",
    "रायो":           "Mustard Greens",
    "तोरी":           "Mustard Greens",
    "मेथी":           "Fenugreek",
    "गुन्द्रुक":      "Gundruk",          # fermented dried greens (unique to Nepal)
    "न्यूरो":         "Fiddlehead Fern",
    "चामसुर":         "Garden Cress",
    "जिरी":           "Garden Cress",
    "सौफ":            "Fennel",
    "सोया":           "Dill",
    "पार्सले":        "Parsley",
    "पुदिना":         "Mint",
    "धनियाँ":         "Coriander",
    "धनिया":          "Coriander",
    "सेलेरी":         "Celery",
    "लेक":            "Leek",
    "लीक":            "Leek",
    "अस्पारागस":      "Asparagus",
    "कुरीलो":         "Asparagus",
    "हरियो प्याज":    "Spring Onion",

    # ── Brassicas ─────────────────────────────────────────────────────────────
    "काउली":          "Cauliflower",
    "बन्दागोभी":      "Cabbage",
    "बन्धागोभी":      "Cabbage",
    "बन्दा":          "Cabbage",
    "ग्यां ठ":        "Knol-Khol",
    "ग्यांठ":         "Knol-Khol",
    "ग्यांठकोबी":     "Knol-Khol",
    "ब्रोकाउली":      "Broccoli",
    "ब्रसेल्स":       "Brussels Sprouts",

    # ── Root vegetables ───────────────────────────────────────────────────────
    "गोलभेडा":        "Tomato",
    "आलु":            "Potato",
    "गाजर":           "Carrot",
    "मूला":           "Radish",
    "मुला":           "Radish",
    "तरुल":           "Yam",
    "सखरखण्ड":        "Sweet Potato",
    "सखरखंड":         "Sweet Potato",
    "पिंडालु":        "Taro",
    "पिन्डालु":       "Taro",
    "चुकन्द्र":       "Beetroot",
    "चुकुन्दर":       "Beetroot",
    "मूल":            "Turnip",

    # ── Alliums ───────────────────────────────────────────────────────────────
    "प्याज":          "Onion",
    "पिँयाज":         "Onion",
    "लसुन":           "Garlic",
    "अदुवा":          "Ginger",

    # ── Gourds ───────────────────────────────────────────────────────────────
    "घिरौला":         "Sponge Gourd",
    "करेला":          "Bitter Gourd",
    "फर्सी":          "Pumpkin",
    "चिचिन्डा":       "Ridge Gourd",
    "चिचिन्डो":       "Ridge Gourd",
    "लौका":           "Bottle Gourd",
    "लौको":           "Bottle Gourd",
    "चित्लाङ":        "Snake Gourd",
    "परवर":           "Pointed Gourd",
    "इस्कुस":         "Chayote",
    "सक्ने":          "Chayote",

    # ── Cucurbits / Others ────────────────────────────────────────────────────
    "काँक्रो":        "Cucumber",
    "काक्रो":         "Cucumber",
    "भन्टा":          "Eggplant",
    "बैंगन":          "Eggplant",
    "भिन्डी":         "Okra",
    "भेन्डी":         "Okra",
    "भिन्डे":         "Okra",
    "भेडे खुर्सानी":  "Bell Pepper",
    "भेडे खु्र्सानी": "Bell Pepper",   # alternate halant spelling
    "क्याप्सिकम":     "Capsicum",
    "खुर्सानी":       "Chilli",
    "मकै":            "Corn",
    "तामा":           "Bamboo Shoot",

    # ── Legumes ───────────────────────────────────────────────────────────────
    "सिमी":           "Beans",
    "राजमा":          "Kidney Beans",
    "केराउ":          "Peas",
    "मटर":            "Peas",
    "बोडी":           "Cowpea",
    "बाकला":          "Broad Beans",
    "भटमास":          "Soybean",

    # ── Mushrooms ─────────────────────────────────────────────────────────────
    "सिताके च्याउ":   "Shiitake Mushroom",
    "राजा च्याउ":     "King Oyster Mushroom",
    "शिताके":         "Shiitake Mushroom",
    "च्याउ":          "Mushroom",
    "ओयस्टर":         "Oyster Mushroom",

    # ── Jackfruit variants ────────────────────────────────────────────────────
    "भुई कटहर":       "Young Jackfruit",    # ground = unripe, eaten as vegetable
    "रुख कटहर":       "Ripe Jackfruit",     # tree = mature fruit

    # ── Fruits ───────────────────────────────────────────────────────────────
    "सुन्तला":        "Orange",
    "जुनार":          "Mandarin Orange",
    "स्याउ":          "Apple",
    "केरा":           "Banana",
    "अम्बा":          "Guava",
    "आँप":            "Mango",
    "कागती":          "Lemon",
    "निबुवा":         "Lime",
    "भुईकटहर":        "Pineapple",
    "अनानास":         "Pineapple",
    "कटहर":           "Jackfruit",
    "खरबुजा":         "Muskmelon",
    "तरबुजा":         "Watermelon",
    "तरबुज":          "Watermelon",
    "अनार":           "Pomegranate",
    "किवि":           "Kiwi",
    "किवी":           "Kiwi",
    "अंगुर":          "Grapes",
    "नासपाती":        "Pear",
    "आरु":            "Peach",
    "आलुबखडा":        "Plum",
    "आलुबखरा":        "Plum",
    "लिची":           "Lychee",
    "ड्र्यागन फ्रुट": "Dragon Fruit",
    "ड्रागन फ्रुट":   "Dragon Fruit",
    "अभोकाडो":        "Avocado",
    "एभोकाडो":        "Avocado",
    "स्ट्रबेरी":      "Strawberry",
    "मेवा":           "Papaya",
    "पपाया":          "Papaya",
    "प्यासन फ्रुट":   "Passion Fruit",
    "नरिवल":          "Coconut",
    "खजुर":           "Dates",
    "लप्सी":          "Lapsi Plum",
    "छ्याप्पी":       "Lapsi Plum",
    "हलुवाबेद":       "Persimmon",
    "अमला":           "Amla (Gooseberry)",
    "आँवला":          "Amla (Gooseberry)",
    "इमली":           "Tamarind",
    "कोइरालो":        "Koiralo (Orchid Tree)",

    # ── Fish / Seafood ────────────────────────────────────────────────────────
    "राहु":           "Rohu Fish",
    "तिलापिया":       "Tilapia",
    "भाकुर":          "Catla Fish",
    "असला":           "Snow Trout",
    "माछा":           "Fish",

    # ── Other ────────────────────────────────────────────────────────────────
    "तोफु":           "Tofu",
    "टोफु":           "Tofu",
    "गुन्द्रुक":      "Gundruk (Dried Greens)",
    "गुन्दुक":        "Gundruk (Dried Greens)",
    "बरेला":          "Ivy Gourd",
    "बकुला":          "Bakula Flower",
    "बकूला":          "Bakula Flower",
    "साँजवन":         "Sajiwan Herb",
    "सजिवन":          "Sajiwan Herb",
    "न्यू":           "Fiddlehead Fern",
    "स्कु":           "Chayote",
    "स्कूस":          "Chayote",

    # ── Spelling variants found in live data ──────────────────────────────────
    "किनु":           "Kinu Mandarin",
    "खु्र्सानी":      "Chilli",      # alternate halant spelling
    "ग्याठ":          "Kohlrabi",
    "चमसूर":          "Garden Cress",
    "चिचिण्डो":       "Ridge Gourd",
    "छ्यापी":         "Lapsi Plum",
    "पालूगो":         "Spinach",
    "पिंडालू":        "Taro",
    "पुदीना":         "Mint",
    "भिण्डी":         "Okra",
    "सेलरी":          "Celery",
}

# ── Variety / cultivar names that appear in parentheses ──────────────────────
# These appear inside (…) in the Nepali name and should be preserved as-is
VARIETY_MAP: dict[str, str] = {
    "फूजी":       "Fuji",
    "झोले":       "Jhole",
    "हाइब्रीड":   "Hybrid",
    "राजमा":      "Rajma",
    "तने":        "Tane",
    "लोकलक्रस":   "Local Cross",
    "लाम्चो":     "Long",
    "डल्लो":      "Round",
    "डल्ले":      "Round",
    "कन्य":       "Kanya",
    "बुलेट":      "Bullet",
    "माछे":       "Maache",
    "अकबरे":      "Akbare",
    "रहु":        "Rohu",
    "बचुवा":      "Bachhuwa",
    "छडी":        "Chhadi",
    "नरिवल":      "Nariwal",
    "ज्यापु":     "Jyapu",
    "चाइनिज":     "Chinese",
    "भारतीय":     "Indian",
    "लोकल":       "Local",
    "तराई":       "Terai",
    "नेपाली":     "Nepali",
    "सुकेको":     "Dried",
    "पाकेको":     "Ripe",
}

# ── Qualifier map ─────────────────────────────────────────────────────────────
QUALIFIER_MAP: dict[str, str] = {
    # Size / shape
    "ठूलो":    "Large",
    "ठुलो":    "Large",
    "सानो":    "Small",
    "लाम्चो":  "Long",
    "डल्लो":   "Round",
    # Origin / variety
    "भारतीय":  "Indian",
    "लोकल":    "Local",
    "स्थानीय": "Local",
    "तराई":    "Terai",
    "हिले":    "Hile",
    "चिनी":    "Chinese",
    "चाइनिज":  "Chinese",
    "हरियो":   "Green",
    "रातो":    "Red",
    "सेतो":    "White",
    "पहेँलो":  "Yellow",
    "पहेंलो":  "Yellow",
    "कालो":    "Black",
    "नेपाली":  "Nepali",
    "ताजा":    "Fresh",
    "सुकेको":  "Dried",
    "सुकेकी":  "Dried",
    "पाकेको":  "Ripe",
    "काँचो":   "Raw",
    "साग":     "Greens",       # leafy green preparation
    "को साग":  "Greens",
}

# ── Unit map ──────────────────────────────────────────────────────────────────
UNIT_MAP: dict[str, str] = {
    "के.जी.":    "KG",
    "केजी":      "KG",
    "कि.ग्रा.":  "KG",
    "किग्रा":    "KG",
    "दर्जन":     "Doz",
    "प्रति गोटा":"Pc",
    "गोटा":      "Pc",
    "ग्राम":     "g",
    "लिटर":      "L",
    "मुठा":      "Bun",
    "के":        "KG",   # catch-all for केजी variants
}


# ── Sorted keys by length descending (most specific wins) ────────────────────
_SORTED_VEG   = sorted(VEGETABLE_MAP.keys(),  key=len, reverse=True)
_SORTED_QUAL  = sorted(QUALIFIER_MAP.keys(),  key=len, reverse=True)
_SORTED_UNIT  = sorted(UNIT_MAP.keys(),       key=len, reverse=True)
_SORTED_VAR   = sorted(VARIETY_MAP.keys(),    key=len, reverse=True)


def translate_name(nepali_name: str) -> str:
    """Translate a Nepali commodity name to English. Returns original if unknown."""
    base_english = None

    # Find best (longest) matching base commodity
    for key in _SORTED_VEG:
        if key in nepali_name:
            base_english = VEGETABLE_MAP[key]
            break

    if base_english is None:
        return nepali_name  # untranslatable — return as-is

    base_key_used = next(k for k in _SORTED_VEG if k in nepali_name)
    qualifiers: list[str] = []

    # 1. Check the full name for qualifier keywords (outside parens)
    for qkey in _SORTED_QUAL:
        if qkey in nepali_name and qkey not in base_key_used:
            en_q = QUALIFIER_MAP[qkey]
            if en_q not in qualifiers:
                qualifiers.append(en_q)

    # 2. Parse parenthetical sections like (भारतीय), (फूजी), (हाइब्रीड)
    for paren in re.findall(r'[（(]([^)）]+)[）)]', nepali_name):
        paren = paren.strip()
        matched = False
        # Check variety map first (more specific)
        for vkey, ven in VARIETY_MAP.items():
            if vkey in paren:
                if ven not in qualifiers:
                    qualifiers.append(ven)
                matched = True
                break
        if not matched:
            # Fall back to qualifier map
            for qkey in _SORTED_QUAL:
                if qkey in paren:
                    en_q = QUALIFIER_MAP[qkey]
                    if en_q not in qualifiers:
                        qualifiers.append(en_q)
                    break

    # Deduplicate while preserving order; remove qualifiers already baked into base
    seen: set[str] = set()
    deduped = []
    for q in qualifiers:
        # Skip if this qualifier word already appears in the base name
        if q not in seen and q.lower() not in base_english.lower():
            seen.add(q)
            deduped.append(q)

    if deduped:
        return f"{base_english} ({', '.join(deduped)})"
    return base_english


def translate_unit(nepali_unit: str) -> str:
    """Translate a Nepali unit string to English abbreviation."""
    normalized = nepali_unit.replace(" ", "").replace(".", "")
    # Normalise the lookup map too
    for k, v in UNIT_MAP.items():
        if k.replace(" ", "").replace(".", "") == normalized:
            return v
    for k, v in UNIT_MAP.items():
        if k in nepali_unit:
            return v
    # Heuristic: contains KG-like Devanagari
    if ("के" in nepali_unit and "जी" in nepali_unit) or "ग्रा" in nepali_unit:
        return "KG"
    return nepali_unit
