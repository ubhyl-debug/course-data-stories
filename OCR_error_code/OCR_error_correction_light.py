import pickle, re
import spacy
from spacy.cli import download as spacy_download
from transformers import pipeline

# 1) Load your merged pickle
with open('merged_all_added.pkl', 'rb') as f:
    df = pickle.load(f)

# 2) Make sure spaCy German is there
try:
    nlp = spacy.load('de_core_news_sm')
except OSError:
    spacy_download('de_core_news_sm')
    nlp = spacy.load('de_core_news_sm')

# 3) Set up your spelling corrector
corrector = pipeline(
    task='text2text-generation',
    model='oliverguhr/spelling-correction-german-base',
    device=-1
)

# optional: small dictionary check fallback
try:
    import enchant
    dict_de = enchant.Dict('de_DE')
    is_valid = lambda w: dict_de.check(w)
except Exception:
    is_valid = lambda w: w.isalpha()

# 4) Grab the first article, take its first 10 sentences
full_text = df.loc[0, 'plainpagefulltext']
doc = nlp(full_text)
first_10_sents = [sent.text for sent in doc.sents][:10]
snippet = " ".join(first_10_sents)
print("=== ORIGINAL SNIPPET ===")
print(snippet)

# 5) Run correction only on that snippet
def correct_text(text):
    corrected = text
    # find all word-tokens
    toks = re.findall(r"\b\w+\b", text)
    for tok in toks:
        # skip if looks valid
        if is_valid(tok):
            continue
        # generate a correction
        prompt = f"Korrigiere das falsch erkannte Wort '{tok}' im deutschen Satz: \"{text}\". Gib nur das Ersatzwort zurück."
        out = corrector(prompt, max_length=16, num_return_sequences=1)
        corr = out[0]['generated_text'].strip() or tok
        # substitute globally
        corrected = re.sub(rf"\b{re.escape(tok)}\b", corr, corrected)
    return corrected

fixed_snippet = correct_text(snippet)
print("\n=== CORRECTED SNIPPET ===")
print(fixed_snippet)