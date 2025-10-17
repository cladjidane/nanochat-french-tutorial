# 🔗 Intégration de Vigostral-7B fine-tuné avec Nanochat

Ce document explique comment intégrer votre modèle Vigostral-7B-Chat fine-tuné avec l'interface de chat de Nanochat.

## 🎯 Objectif

Utiliser l'application complète **nanochat** (interface web, CLI, outils comme calculatrice) mais avec votre modèle conversationnel français personnalisé au lieu du modèle anglais par défaut.

## ⚠️ Le Défi Technique

Il y a un problème d'**incompatibilité d'architecture** :

| Aspect | Nanochat | Vigostral-7B-Chat |
|--------|----------|-------------------|
| **Architecture** | GPT custom (nanochat/gpt.py) | Mistral-7B (architecture Llama) |
| **Format checkpoints** | `.pt` PyTorch natif | HuggingFace format |
| **Tokenizer** | BPE custom (rustbpe) | Tokenizer Llama/Mistral |
| **Position encoding** | RoPE custom | RoPE standard |
| **Attention** | Multi-Query Attention (MQA) | Grouped-Query Attention (GQA) |

**Conclusion** : Nanochat ne peut PAS charger directement Vigostral car ce sont deux architectures complètement différentes.

## 💡 Solutions Possibles

### Option A : Créer un wrapper/adapter pour nanochat

**Principe** : Modifier l'interface de nanochat pour qu'elle puisse charger des modèles HuggingFace comme Vigostral.

**Avantages** :
- ✅ Garde l'interface utilisateur de nanochat (web, CLI)
- ✅ Utilise votre modèle Vigostral fine-tuné
- ✅ Peut réutiliser les outils de nanochat (calculatrice, etc.)

**Inconvénients** :
- ❌ Nécessite de modifier le code de nanochat
- ❌ Doit créer une couche d'abstraction entre nanochat et HuggingFace
- ❌ Maintenance complexe si nanochat évolue

**Fichiers à modifier** :
```
nanochat/
├── checkpoint_manager.py  # Ajouter support HuggingFace
├── engine.py              # Adapter la génération pour Transformers
└── scripts/
    ├── chat_cli.py        # Détecter type de modèle
    └── chat_web.py        # Idem
```

**Exemple de code** (checkpoint_manager.py) :
```python
def load_model(source, device, phase, model_tag=None, step=None):
    if source.startswith("hf_"):
        # Charger depuis HuggingFace
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        model_name = source.replace("hf_", "")
        base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        # Si des adaptateurs LoRA existent, les charger
        if os.path.exists(f"./models/{model_name}_lora"):
            model = PeftModel.from_pretrained(base_model, f"./models/{model_name}_lora")
        else:
            model = base_model

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer, {}
    else:
        # Logique nanochat existante
        # ...
```

**Note** : Le nom correct du modèle Vigostral est `bofenghuang/vigostral-7b-chat` (pas `vigostral/vigostral-7b-chat`).

---

### Option B : Fork nanochat et créer nanochat-french

**Principe** : Créer une version standalone de nanochat qui utilise exclusivement des modèles HuggingFace français.

**Avantages** :
- ✅ Code clean et dédié aux modèles français
- ✅ Peut simplifier beaucoup le code (pas besoin de GPT custom)
- ✅ Facile à maintenir indépendamment

**Inconvénients** :
- ❌ Perd les synchronisations avec nanochat upstream
- ❌ Doit réimplémenter certaines fonctionnalités

**Structure proposée** :
```
nanochat-french/
├── models/
│   └── vigostral-finetuned/  # Votre modèle
├── chat_cli.py                # CLI simplifié pour Vigostral
├── chat_web.py                # Interface web
└── engine.py                  # Wrapper Transformers
```

---

### Option C : Interface web standalone avec Gradio

**Principe** : Créer une interface web simple avec Gradio, sans réutiliser nanochat.

**Avantages** :
- ✅ Très simple et rapide à implémenter
- ✅ Interface moderne et jolie
- ✅ Aucune dépendance à nanochat

**Inconvénients** :
- ❌ Perd les outils spécifiques de nanochat (calculatrice, etc.)
- ❌ Moins de contrôle sur l'interface

**Code complet** (~50 lignes) :
```python
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Charger le modèle
model_name = "vigostral/vigostral-7b-chat"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model = PeftModel.from_pretrained(model, "./vigostral-finetuned-final")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def chat(message, history):
    # Formater le prompt
    prompt = f"<s>[INST] {message} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Générer
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("[/INST]")[1].strip()

    return response

# Interface Gradio
demo = gr.ChatInterface(
    chat,
    title="🇫🇷 Chat Français Personnalisé",
    description="Modèle Vigostral-7B fine-tuné sur votre style",
)

demo.launch(share=True)
```

---

## 🎯 Recommandation : Option C (Gradio)

Pour commencer, je recommande **l'Option C** car :

1. **Simple** : 50 lignes de code, fonctionne immédiatement
2. **Pas de modifications complexes** : Pas besoin de forker nanochat
3. **Interface moderne** : Gradio est très joli et facile à utiliser
4. **Partage facile** : `share=True` génère un URL public temporaire

### Étapes pour implémenter l'Option C

1. **Fine-tuner Vigostral sur Colab** (notebook fourni : `vigostral_finetune_colab.ipynb`)
2. **Télécharger le modèle fine-tuné** (~100-200 MB d'adaptateurs LoRA)
3. **Créer l'interface Gradio localement** (voir code ci-dessus)
4. **Lancer** : `python chat_gradio.py`

---

## 📊 Comparaison des Options

| Critère | Option A (Wrapper) | Option B (Fork) | Option C (Gradio) |
|---------|-------------------|-----------------|-------------------|
| **Complexité** | 🔴 Élevée | 🟡 Moyenne | 🟢 Faible |
| **Temps d'implémentation** | 2-3 jours | 1-2 jours | 1-2 heures |
| **Interface** | Interface nanochat | Interface nanochat | Interface Gradio |
| **Outils (calculatrice)** | ✅ Oui | ⚠️ À réimplémenter | ❌ Non (facilement ajoutable) |
| **Maintenance** | 🔴 Difficile | 🟡 Moyenne | 🟢 Simple |
| **Compatibilité future** | ⚠️ Dépend de nanochat | ✅ Indépendant | ✅ Indépendant |

---

## 🚀 Prochaines Étapes

### Phase 1 : Fine-tuning (Maintenant)
1. ✅ Utiliser `vigostral_finetune_colab.ipynb` sur Google Colab
2. ✅ Fine-tuner Vigostral-7B sur vos 123 dialogues avec LoRA
3. ✅ Télécharger le modèle fine-tuné

### Phase 2 : Interface locale (Après fine-tuning)
1. Installer les dépendances localement :
   ```bash
   pip install transformers peft accelerate bitsandbytes gradio
   ```

2. Créer `chat_gradio.py` avec le code fourni ci-dessus

3. Lancer l'interface :
   ```bash
   python chat_gradio.py
   ```

### Phase 3 : Améliorations (Optionnel)
- Ajouter des outils (calculatrice, recherche web)
- Créer une vraie intégration avec nanochat (Option A ou B)
- Déployer en ligne (HuggingFace Spaces gratuit)

---

## 💡 Notes Techniques Importantes

### Chargement du modèle en local

Si vous avez une **machine avec GPU** (CUDA) :
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

Si vous êtes sur **CPU/Mac** :
```python
# Utiliser quantization 4-bit pour réduire la mémoire
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float32,  # CPU = float32
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cpu"
)
```

**Attention** : Sur CPU, la génération sera **lente** (~1-2 tokens/seconde pour 7B params).

### Taille du modèle

- **Modèle de base (Vigostral-7B)** : ~14 GB (FP16) ou ~4 GB (4-bit)
- **Adaptateurs LoRA fine-tunés** : ~100-200 MB
- **Total avec LoRA** : ~14.2 GB (FP16) ou ~4.2 GB (4-bit)

---

## 📚 Ressources Complémentaires

- [Documentation Gradio](https://gradio.app/docs/)
- [Documentation PEFT (LoRA)](https://huggingface.co/docs/peft)
- [Vigostral-7B-Chat sur HuggingFace](https://huggingface.co/vigostral/vigostral-7b-chat)
- [Tutoriel LoRA officiel](https://huggingface.co/blog/lora)

---

## ❓ FAQ

### Q : Puis-je utiliser Vigostral fine-tuné avec nanochat sans modifications ?
**R** : Non, nanochat utilise une architecture GPT custom incompatible avec Vigostral (architecture Mistral). Il faut soit créer un wrapper (Option A), soit utiliser une interface standalone (Option C).

### Q : Est-ce que LoRA réduit la qualité du modèle ?
**R** : Non ! Des études montrent que LoRA donne des résultats comparables au fine-tuning complet, tout en n'entraînant que ~1% des paramètres.

### Q : Combien de temps prend le fine-tuning sur Colab ?
**R** : Environ **20-30 minutes** sur un GPU T4 gratuit pour 123 dialogues avec 3 epochs.

### Q : Puis-je fine-tuner sur plus de dialogues ?
**R** : Oui ! Plus vous avez de dialogues de qualité (200-500+), meilleur sera le résultat. Le temps d'entraînement augmente proportionnellement.

### Q : Le modèle fine-tuné fonctionne-t-il sur Mac/CPU ?
**R** : Oui, mais ce sera **très lent** (~1-2 tokens/seconde). Pour une expérience fluide, utilisez un GPU (T4, RTX 3060, M1/M2 Mac avec MPS).

---

**Made with ❤️ for the French AI community**

Pour toute question : [Issues GitHub](https://github.com/cladjidane/nanochat-french-tutorial/issues)
