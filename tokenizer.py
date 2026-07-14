# tokenizer.py - Le traducteur de texte en nombres pour ChadLLM

import json
import re
from pathlib import Path

class Tokenizer:
    """
    Le Tokenizer est le traducteur de notre IA.
    Il transforme le texte humain en nombres que le cerveau peut comprendre.
    """
    
    def __init__(self):
        # Le vocabulaire : un dictionnaire qui associe chaque mot à un numéro
        # Exemple: {"le": 1, "cheval": 2, "est": 3, ...}
        self.vocab = {}
        
        # L'inverse du vocabulaire : pour retrouver le mot à partir du numéro
        # Exemple: {1: "le", 2: "cheval", 3: "est", ...}
        self.inverse_vocab = {}
        
        # Les tokens spéciaux
        self.pad_token = "[PAD]"   # Remplissage (pour compléter les phrases courtes)
        self.unk_token = "[UNK]"   # Inconnu (pour les mots qu'on n'a jamais vus)
        self.bos_token = "[BOS]"   # Début de phrase (Begin Of Sentence)
        self.eos_token = "[EOS]"   # Fin de phrase (End Of Sentence)
        
        # On initialise le vocabulaire avec les tokens spéciaux
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
    
    def nettoyer_texte(self, texte):
        """
        Nettoie le texte : met en minuscule, enlève les espaces en trop
        """
        # Mettre en minuscule
        texte = texte.lower()
        
        # Remplacer les multiples espaces par un seul
        texte = re.sub(r'\s+', ' ', texte)
        
        # Enlever les espaces au début et à la fin
        texte = texte.strip()
        
        return texte
    
    def decouper_en_mots(self, texte):
        """
        Découpe le texte en mots (tokenization simple)
        On sépare sur les espaces et la ponctuation
        """
        # Ajouter des espaces autour de la ponctuation
        # "Bonjour!" devient "Bonjour !"
        texte = re.sub(r'([.!?,;:])', r' \1 ', texte)
        
        # Découper sur les espaces
        mots = texte.split()
        
        return mots
    
    def construire_vocabulaire(self, chemin_corpus):
        """
        Lit le fichier corpus.txt et construit le vocabulaire
        """
        print(f"📖 Lecture du corpus : {chemin_corpus}")
        
        # Lire le fichier
        with open(chemin_corpus, 'r', encoding='utf-8') as f:
            lignes = f.readlines()
        
        print(f"✅ {len(lignes)} phrases trouvées")
        
        # Parcourir chaque ligne
        for ligne in lignes:
            # Nettoyer la ligne
            texte_propre = self.nettoyer_texte(ligne)
            
            # Découper en mots
            mots = self.decouper_en_mots(texte_propre)
            
            # Ajouter chaque mot au vocabulaire s'il n'existe pas déjà
            for mot in mots:
                if mot not in self.vocab:
                    # Donner un nouveau numéro à ce mot
                    nouveau_numero = len(self.vocab)
                    self.vocab[mot] = nouveau_numero
                    self.inverse_vocab[nouveau_numero] = mot
        
        print(f"🎉 Vocabulaire construit : {len(self.vocab)} mots uniques !")
    
    def encoder(self, texte):
        """
        Transforme une phrase en liste de nombres
        Exemple: "le cheval" → [4, 5]
        """
        # Nettoyer et découper
        texte_propre = self.nettoyer_texte(texte)
        mots = self.decouper_en_mots(texte_propre)
        
        # Convertir chaque mot en son numéro
        numeros = []
        for mot in mots:
            if mot in self.vocab:
                # Le mot existe dans notre vocabulaire
                numeros.append(self.vocab[mot])
            else:
                # Le mot n'existe pas, on utilise le token [UNK] (inconnu)
                numeros.append(self.vocab[self.unk_token])
        
        return numeros
    
    def decoder(self, numeros):
        """
        Transforme une liste de nombres en phrase
        Exemple: [4, 5] → "le cheval"
        """
        mots = []
        for numero in numeros:
            if numero in self.inverse_vocab:
                mots.append(self.inverse_vocab[numero])
            else:
                mots.append(self.unk_token)
        
        # Recoller les mots avec des espaces
        texte = ' '.join(mots)
        
        return texte
    
    def sauvegarder(self, chemin_sauvegarde):
        """
        Sauvegarde le vocabulaire dans un fichier JSON
        Pour ne pas avoir à le reconstruire à chaque fois
        """
        donnees = {
            'vocab': self.vocab,
            'inverse_vocab': {str(k): v for k, v in self.inverse_vocab.items()}
        }
        
        with open(chemin_sauvegarde, 'w', encoding='utf-8') as f:
            json.dump(donnees, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Vocabulaire sauvegardé : {chemin_sauvegarde}")
    
    def charger(self, chemin_sauvegarde):
        """
        Charge un vocabulaire déjà sauvegardé
        """
        with open(chemin_sauvegarde, 'r', encoding='utf-8') as f:
            donnees = json.load(f)
        
        self.vocab = donnees['vocab']
        self.inverse_vocab = {int(k): v for k, v in donnees['inverse_vocab'].items()}
        
        print(f"📂 Vocabulaire chargé : {len(self.vocab)} mots")


# ===== TEST DU TOKENIZER =====
if __name__ == "__main__":
    print("🚀 Test du Tokenizer ChadLLM\n")
    
    # Créer le tokenizer
    tokenizer = Tokenizer()
    
    # Créer un petit corpus de test
    corpus_test = "data/corpus_test.txt"
    Path("data").mkdir(exist_ok=True)
    
    with open(corpus_test, 'w', encoding='utf-8') as f:
        f.write("le cheval est magnifique\n")
        f.write("un beau cheval blanc\n")
        f.write("le cheval noir galope\n")
        f.write("j'aime les chevaux\n")
    
    print("📝 Corpus de test créé\n")
    
    # Construire le vocabulaire
    tokenizer.construire_vocabulaire(corpus_test)
    
    # Sauvegarder le vocabulaire
    tokenizer.sauvegarder("data/vocabulaire.json")
    
    print("\n" + "="*50)
    print("🧪 TESTS\n")
    
    # Test 1 : Encoder une phrase
    phrase_test = "le cheval est beau"
    numeros = tokenizer.encoder(phrase_test)
    print(f"📝 Phrase : '{phrase_test}'")
    print(f"🔢 Nombres : {numeros}\n")
    
    # Test 2 : Décoder des nombres
    texte_decode = tokenizer.decoder(numeros)
    print(f"🔢 Nombres : {numeros}")
    print(f"📝 Texte décodé : '{texte_decode}'\n")
    
    # Test 3 : Mot inconnu
    phrase_inconnue = "le cheval mange une pomme"
    numeros_inconnus = tokenizer.encoder(phrase_inconnue)
    print(f"📝 Phrase avec mot inconnu : '{phrase_inconnue}'")
    print(f"🔢 Nombres : {numeros_inconnus}")
    print(f"💡 'pomme' n'est pas dans le vocabulaire, donc [UNK] est utilisé\n")
    
    print("="*50)
    print("✅ Tokenizer prêt à l'emploi !")