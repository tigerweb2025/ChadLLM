import torch
import torch.nn as nn
from tokenizer import Tokenizer
from model import ChadLLM

def preparer_donnees(tokenizer, chemin_corpus, longueur_seq=12):
    with open(chemin_corpus, 'r', encoding='utf-8') as f:
        texte = f.read()
    tokens = tokenizer.encoder(texte)
    X, Y = [], []
    for i in range(len(tokens) - longueur_seq):
        X.append(tokens[i : i+longueur_seq])
        Y.append(tokens[i+1 : i+longueur_seq+1])
    return torch.tensor(X), torch.tensor(Y)

def entrainer_modele():
    print("🚀 Démarrage de l'entraînement V2 de ChadLLM...\n")

    tokenizer = Tokenizer()
    tokenizer.charger("data/vocabulaire.json")
    taille_vocab = len(tokenizer.vocab)

    # PARAMÈTRES EXACTS POUR L'ENTRAÎNEMENT
    modele = ChadLLM(
        taille_vocab=taille_vocab,
        taille_embedding=128,
        nombre_tetes=8,
        nombre_couches=4
    )

    X, Y = preparer_donnees(tokenizer, "data/corpus.txt", longueur_seq=12)
    print(f"📚 {len(X)} séquences d'entraînement prêtes.\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(modele.parameters(), lr=0.001)

    epochs = 40
    batch_size = 32 
    
    for epoch in range(epochs):
        epoch_loss = 0
        indices = torch.randperm(len(X)) 
        
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]
            
            predictions = modele(X_batch)
            loss = criterion(predictions.view(-1, taille_vocab), Y_batch.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"🔄 Époque {epoch+1:02d} | Erreur (Loss): {epoch_loss / (len(X)//batch_size):.4f} 📉")

    print("\n✅ Entraînement terminé !")
    torch.save(modele.state_dict(), "data/poids_modele.pth")
    print("💾 Poids sauvegardés")

if __name__ == "__main__":
    entrainer_modele()