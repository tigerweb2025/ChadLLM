import random

def generer():
    print("📝 Génération du corpus structuré (Dialogues Q/R)...")
    
    # Paires de dialogues (Question / Réponse)
    dialogues = [
        ("bonjour", "salut comment vas-tu"),
        ("salut", "bonjour ça va bien et toi"),
        ("coucou", "hey salut toi"),
        ("comment vas-tu", "je vais très bien merci"),
        ("ça va", "oui ça va super et toi"),
        ("tu fais quoi", "je discute avec toi c'est sympa"),
        ("quel est ton nom", "je m'appelle chadllm"),
        ("comment tu t'appelles", "mon nom est chadllm"),
        ("qui es-tu", "je suis une intelligence artificielle"),
        ("tu es qui", "je suis ton assistant chadllm"),
        ("le ciel est", "bleu quand il fait beau"),
        ("le soleil est", "chaud et lumineux aujourd'hui"),
        ("l'eau est", "essentielle pour la vie"),
        ("la terre est", "notre belle planète"),
        ("parle-moi des chevaux", "les chevaux sont de magnifiques animaux"),
        ("tu aimes les chevaux", "oui j'adore les chevaux ils sont élégants"),
        ("un cheval blanc", "est un animal très majestueux"),
        ("un cheval noir", "est superbe et puissant"),
        ("que sais-tu faire", "je sais discuter et apprendre de toi"),
        ("tu es intelligent", "je fais de mon mieux pour m'améliorer"),
        ("tu comprends", "oui je comprends le français parfaitement"),
        ("merci beaucoup", "de rien c'est avec grand plaisir"),
        ("merci", "je t'en prie"),
        ("au revoir", "à bientôt passe une bonne journée"),
        ("bonne journée", "merci toi aussi à la prochaine"),
        ("à bientôt", "oui on se reparle très vite"),
        ("c'était sympa", "oui j'ai beaucoup aimé discuter avec toi"),
        ("tu es là", "oui je suis toujours là pour toi"),
        ("il fait beau", "oui le soleil brille fort aujourd'hui"),
        ("j'aime les animaux", "moi aussi surtout les chevaux et les chiens"),
    ]
    
    corpus = []
    for _ in range(3000):
        echanges = random.sample(dialogues, random.randint(2, 4))
        for question, reponse in echanges:
            corpus.append(question)
            corpus.append(reponse)
        corpus.append("") 
    
    with open("data/corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(corpus))
    print(f"✅ Corpus structuré créé ({len(corpus)} lignes)")

if __name__ == "__main__":
    generer()