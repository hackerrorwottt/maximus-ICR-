from spellchecker import SpellChecker
import re
import torch

def postprocess_text(text):
    """
    Basic post-processing to clean up common OCR/ICR mistakes.
    We use pyspellchecker to find misspelled words and suggest corrections.
    """
    if not text.strip():
        return text

    # Initialize spell checker
    spell = SpellChecker()
    
    # Simple tokenization: split by spaces and keep punctuation separate if possible
    # We'll use a basic regex to find words (alphanumeric sequences)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    # Find those words that may be misspelled.
    # Note on Names/Addresses: Spellcheckers will ruthlessly "correct" names (like "Aarav") 
    # to the closest dictionary word. For a learning project, this shows how post-processing 
    # can sometimes hallucinate errors. We sort the set to ensure deterministic iteration.
    misspelled = sorted(list(spell.unknown(words)))
    
    corrected_text = text
    
    for word in misspelled:
        # Simple heuristic to avoid wrecking names: skip capitalized words.
        # Tradeoff: This will also skip the first word of every sentence, meaning 
        # genuinely misspelled words at the start of a sentence won't be corrected.
        # For a learning project, this is a safer tradeoff than mangling proper nouns.
        if word.istitle():
            continue
            
        # Get the one `most likely` answer
        correction = spell.correction(word)
        if correction and correction != word:
            # Note: This simple replacement might replace substrings if we aren't careful,
            # but for a learning project, regex word boundary replacement works well.
            # We use \b to ensure we only replace whole words.
            pattern = r'\b' + re.escape(word) + r'\b'
            corrected_text = re.sub(pattern, correction, corrected_text)
            
    # Additional common OCR cleanups can go here:
    # Remove isolated periods or commas that are often hallucinated by TrOCR crops
    corrected_text = re.sub(r'[ \t]+([.,;:?!])[ \t]+', ' ', corrected_text)
    
    # Replacing multiple spaces with a single space, preserving newlines
    corrected_text = re.sub(r'[ \t]+', ' ', corrected_text)
    
    # Remove excessive blank lines
    corrected_text = re.sub(r'\n{3,}', '\n\n', corrected_text)
    
    return corrected_text.strip()


