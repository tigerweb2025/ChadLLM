import torch
from tokenizer import Tokenizer
from model import ChadLLM

def discuter_avec_lia():
    print("🤖 Chargement de ChadLLM V2...\n")

    tokenizer = Tokenizer()
    tokenizer.charger("data/vocabulaire.json")
    taille_vocab = len(tokenizer.vocab)

    # PARAMÈTRES EXACTS POUR LA GÉNÉRATION
    modele = ChadLLM(
        taille_vocab=taille_vocab,
        taille_embedding=128,
        nombre_tetes=8,
        nombre_couches=4
    )
    modele.load_state_dict(torch.load("data/poids_modele.pth", weights_only=True))
    modele.eval()

    print("✅ ChadLLM V2 est prêt ! Tape 'quitter' pour arrêter.\n")
    print("="*50)

    while True:
        texte_utilisateur = input("👤 Toi : ")
        if texte_utilisateur.lower() == 'quitter':
            break
        if not texte_utilisateur.strip():
            continue

        tokens = tokenizer.encoder(texte_utilisateur)
        tokens_tensor = torch.tensor([tokens])

        with torch.no_grad():
            # TEMPÉRATURE 0.5 POUR RÉPONSES STRICTES
            tokens_generes = modele.generer(tokens_tensor, longueur_max_generation=15, temperature=0.5)

        texte_genere = tokenizer.decoder(tokens_generes[0].tolist())
        texte_propre = texte_genere.replace("[BOS]", "").replace("[EOS]", "").replace("[PAD]", "").replace("[UNK]", "").strip()
        
        reponse = texte_propre.replace(texte_utilisateur.lower(), "", 1).strip()
        
        print(f"🤖 ChadLLM : {reponse}\n")

if __name__ == "__main__":
    discuter_avec_lia()