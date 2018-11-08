import re
from utils.contractions import CONTRACTION_MAP


def remove_tags(text):
    tag = re.compile(r'<.+>')
    processed_text = tag.sub("", text)
    return processed_text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    # Function used with reference from https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_punct(text):
    punct = re.compile(r'[^\w ]')
    text = punct.sub("", text)
    return text

def preprocess_text(text, lower=True, tags=True, contractions=True, strip_punct=True):
    if lower:
        text = text.lower()
    if tags:
        text = remove_tags(text)
    if contractions:
        text = expand_contractions(text)
    if strip_punct:
        text = remove_punct(text)

    return text
