import re

# Table de suppression des voyelles et signes diacritiques (tashkeel)
_TASHKEEL_REGEX = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]')

# Tatweel (kashida)
_TATWEEL_REGEX = re.compile(r'\u0640')

def remove_tashkeel(text: str) -> str:
    """Supprime tous les diacritiques arabes (Fatha, Damma, Kasra, Sukun, Tanwin, Shadda, etc.)."""
    if not text:
        return ""
    return _TASHKEEL_REGEX.sub('', text)

def remove_tatweel(text: str) -> str:
    """Supprime le caractère tatweel (kashida)."""
    if not text:
        return ""
    return _TATWEEL_REGEX.sub('', text)

def normalize_arabic(text: str) -> str:
    """
    Normalisation standardisée pour les mots d'entrée du lemmatiseur :
    1. Suppression du tatweel.
    2. Suppression du tashkeel.
    3. Normalisation des variantes d'Alif (أ, إ, آ, ٱ -> ا).
    4. Normalisation de l'Alif Maqsura (ى -> ي) si nécessaire ou préservation.
    5. Normalisation de Ta Marbuta (ة -> ه) si standardisé.
    """
    if not text:
        return ""
    text = remove_tatweel(text)
    text = remove_tashkeel(text)
    
    # Normalisation des Alifs
    text = re.sub(r'[إأآٱ]', 'ا', text)
    
    # Normalisation Ta Marbuta et Alif Maqsura optionnelle (standard NLP arabe)
    # Dans les travaux de référence, les Alifs sont unifiés, et on conserve la distinction de base.
    text = text.strip()
    return text
