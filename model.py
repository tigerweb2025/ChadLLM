import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EmbeddingsAvecPosition(nn.Module):
    def __init__(self, taille_vocab, taille_embedding, longueur_max):
        super().__init__()
        self.token_embedding = nn.Embedding(taille_vocab, taille_embedding)
        self.position_embedding = nn.Embedding(longueur_max, taille_embedding)
        self.taille_embedding = taille_embedding
        
    def forward(self, x):
        longueur_sequence = x.size(1)
        positions = torch.arange(longueur_sequence, device=x.device).unsqueeze(0)
        tokens_emb = self.token_embedding(x) * math.sqrt(self.taille_embedding)
        positions_emb = self.position_embedding(positions)
        return tokens_emb + positions_emb

class AttentionTeteUnique(nn.Module):
    def __init__(self, taille_embedding):
        super().__init__()
        self.query = nn.Linear(taille_embedding, taille_embedding, bias=False)
        self.key = nn.Linear(taille_embedding, taille_embedding, bias=False)
        self.value = nn.Linear(taille_embedding, taille_embedding, bias=False)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        masque = torch.triu(torch.ones(scores.size(-2), scores.size(-1), device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(masque == 1, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        return torch.matmul(attention, V)

class AttentionMultiTetes(nn.Module):
    def __init__(self, taille_embedding, nombre_tetes):
        super().__init__()
        assert taille_embedding % nombre_tetes == 0
        self.nombre_tetes = nombre_tetes
        self.taille_tete = taille_embedding // nombre_tetes
        self.tetes = nn.ModuleList([AttentionTeteUnique(self.taille_tete) for _ in range(nombre_tetes)])
        self.projection_finale = nn.Linear(taille_embedding, taille_embedding)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.nombre_tetes, self.taille_tete).transpose(1, 2)
        x_split = x.chunk(self.nombre_tetes, dim=1)
        sorties_tetes = [tete(x_split[i].squeeze(1)) for i, tete in enumerate(self.tetes)]
        sortie_concat = torch.cat(sorties_tetes, dim=-1).view(batch_size, seq_len, -1)
        return self.projection_finale(sortie_concat)

class ReseauFeedForward(nn.Module):
    def __init__(self, taille_embedding, taille_cachee):
        super().__init__()
        self.lineaire1 = nn.Linear(taille_embedding, taille_cachee)
        self.lineaire2 = nn.Linear(taille_cachee, taille_embedding)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.lineaire2(self.activation(self.lineaire1(x)))

class BlocTransformer(nn.Module):
    def __init__(self, taille_embedding, nombre_tetes, taille_cachee):
        super().__init__()
        self.attention = AttentionMultiTetes(taille_embedding, nombre_tetes)
        self.feed_forward = ReseauFeedForward(taille_embedding, taille_cachee)
        self.norm1 = nn.LayerNorm(taille_embedding)
        self.norm2 = nn.LayerNorm(taille_embedding)
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class ChadLLM(nn.Module):
    # PARAMÈTRES EXACTS PROMIS ICI
    def __init__(self, taille_vocab, taille_embedding=128, nombre_tetes=8, nombre_couches=4, longueur_max=128):
        super().__init__()
        self.taille_embedding = taille_embedding
        self.longueur_max = longueur_max
        self.embeddings = EmbeddingsAvecPosition(taille_vocab, taille_embedding, longueur_max)
        self.dropout = nn.Dropout(0.1)
        self.blocs_transformer = nn.ModuleList([
            BlocTransformer(taille_embedding, nombre_tetes, taille_embedding * 4) 
            for _ in range(nombre_couches)
        ])
        self.norm_finale = nn.LayerNorm(taille_embedding)
        self.tete_sortie = nn.Linear(taille_embedding, taille_vocab, bias=False)
        self.apply(self._initialiser_poids)
    
    def _initialiser_poids(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.dropout(self.embeddings(x))
        for bloc in self.blocs_transformer:
            x = bloc(x)
        return self.tete_sortie(self.norm_finale(x))
    
    # TEMPÉRATURE EXACTE PROMISE ICI (0.5)
    def generer(self, tokens_initiaux, longueur_max_generation=20, temperature=0.5):
        self.eval()
        tokens = tokens_initiaux.clone()
        with torch.no_grad():
            for _ in range(longueur_max_generation):
                if tokens.size(1) > self.longueur_max:
                    tokens = tokens[:, -self.longueur_max:]
                logits = self.forward(tokens)[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                prochain_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, prochain_token], dim=1)
        return tokens