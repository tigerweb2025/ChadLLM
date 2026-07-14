from tokenizer import Tokenizer

print("📖 Lecture du nouveau corpus et création du vocabulaire...")
tokenizer = Tokenizer()
tokenizer.construire_vocabulaire("data/corpus.txt")
tokenizer.sauvegarder("data/vocabulaire.json")