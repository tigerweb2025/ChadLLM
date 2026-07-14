from flask import Flask, render_template, request, jsonify
import torch
from tokenizer import Tokenizer
from model import ChadLLM

app = Flask(__name__)

# Charger le modèle une seule fois au démarrage
print("🤖 Chargement de ChadLLM...")
tokenizer = Tokenizer()
tokenizer.charger("data/vocabulaire.json")
taille_vocab = len(tokenizer.vocab)

modele = ChadLLM(
    taille_vocab=taille_vocab,
    taille_embedding=128,
    nombre_tetes=8,
    nombre_couches=4
)
modele.load_state_dict(torch.load("data/poids_modele.pth", weights_only=True))
modele.eval()
print("✅ ChadLLM prêt !\n")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'response': ''})
    
    # Encoder le message
    tokens = tokenizer.encoder(message)
    tokens_tensor = torch.tensor([tokens])
    
    # Générer la réponse
    with torch.no_grad():
        tokens_generes = modele.generer(
            tokens_tensor, 
            longueur_max_generation=15, 
            temperature=0.5
        )
    
    # Décoder la réponse
    texte_genere = tokenizer.decoder(tokens_generes[0].tolist())
    texte_propre = texte_genere.replace("[BOS]", "").replace("[EOS]", "").replace("[PAD]", "").replace("[UNK]", "").strip()
    
    # Enlever le message de l'utilisateur pour n'afficher que la réponse
    reponse = texte_propre.replace(message.lower(), "", 1).strip()
    
    return jsonify({'response': reponse})

if __name__ == '__main__':
    print("="*50)
    print("🌐 ChadLLM Web Interface")
    print("📍 Ouvre ton navigateur et va sur : http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)