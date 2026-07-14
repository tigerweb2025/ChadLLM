markdown
# 🤖 ChadLLM - Mon Premier LLM From Scratch

**ChadLLM** est un mini-modèle de langage (LLM) construit entièrement de zéro (from scratch) à des fins pédagogiques.

Ce projet démontre comment fonctionne une intelligence artificielle sous le capot, sans utiliser de modèles pré-entraînés.

##  Fonctionnalités

- 🧠 **Architecture Transformer personnalisée** (Embeddings, Multi-Head Attention, Feed-Forward)
- 📖 **Tokenizer sur mesure** pour le français
- 🌐 **Interface Web interactive** (style ChatGPT) construite avec Flask
- ⚙️ **Entraînement local** sur CPU avec PyTorch

## 🛠️ Prérequis

- Python 3.11 ou supérieur
- Git

## 🚀 Installation et Lancement

### 1. Cloner le projet

```bash
git clone https://github.com/tigerweb2025/ChadLLM.git
cd ChadLLM
```

### 2. Créer l'environnement virtuel
```bash
python -m venv venv
```

**Sur Windows (PowerShell) :**

```powershell
.\venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install flask numpy
```

## 🎯 Utilisation

### 1. Générer le corpus d'entraînement

```bash
python creer_corpus.py
```

### 2. Construire le vocabulaire (Tokenizer)

```bash
python preparer.py
```

### 3. Entraîner le modèle

(Cela peut prendre quelques minutes sur un CPU)

```bash
python train.py
```

### 4. Lancer l'interface Web

```bash
python app.py
```

Ouvrez ensuite votre navigateur et allez sur : **http://127.0.0.1:5000**

## 📁 Structure du projet

```
ChadLLM/
├── app.py                 # Serveur Flask (Interface Web)
├── model.py               # Cerveau du Transformer (From Scratch)
├── tokenizer.py           # Traducteur Texte <-> Nombres
├── train.py               # Script d'entraînement (Backpropagation)
├── generate.py            # Script de génération en terminal
├── creer_corpus.py        # Générateur de données
├── preparer.py            # Création du vocabulaire
├── templates/
│   └── index.html         # Interface utilisateur (Frontend)
└── data/
    ├── corpus.txt         # Texte d'entraînement
    └── vocabulaire.json   # Vocabulaire sauvegardé
```

---

## Créé avec passion - Juillet 2026

