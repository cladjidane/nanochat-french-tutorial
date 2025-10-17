# 🇫🇷 Fine-tuning d'un Modèle Conversationnel Français

Tutorial complet pour fine-tuner **Vigostral-7B-Chat**, un modèle conversationnel français de 7 milliards de paramètres, sur vos propres dialogues avec **Google Colab** (GPU gratuit) et **LoRA** (Low-Rank Adaptation).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cladjidane/nanochat-french-tutorial/blob/main/vigostral_finetune_colab.ipynb)

## 🎯 Objectif

Créer un chatbot en français qui parle dans **votre style personnel** en fine-tunant Vigostral-7B-Chat sur vos conversations.

**Ce que vous obtiendrez** :
- ✅ Un modèle conversationnel français performant
- ✅ Fine-tuné sur vos dialogues personnels (votre style, votre vocabulaire)
- ✅ Utilisable via une interface web moderne (Gradio)
- ✅ Entraînement rapide (~20-30 minutes sur GPU T4 gratuit)
- ✅ Fonctionne localement après téléchargement

## 🚀 Démarrage Rapide

### Étape 1 : Fine-tuning sur Google Colab (20-30 minutes)

1. Cliquez sur le badge "Open in Colab" ci-dessus
2. **Runtime** → **Change runtime type** → Sélectionnez **T4 GPU**
3. Uploadez votre dataset `combined_dataset.jsonl` quand demandé
4. Exécutez toutes les cellules du notebook
5. Téléchargez le modèle fine-tuné à la fin

### Étape 2 : Interface locale (5 minutes)

1. Installez les dépendances :
   ```bash
   pip install transformers peft accelerate bitsandbytes gradio torch
   ```

2. Lancez l'interface web :
   ```bash
   python chat_gradio.py
   ```

3. Ouvrez votre navigateur sur `http://localhost:7860` 🎉

## 📊 Format du dataset

Votre fichier `combined_dataset.jsonl` doit contenir des dialogues au format OpenAI :

```jsonl
{"messages": [{"role": "user", "content": "Bonjour, comment vas-tu ?"}, {"role": "assistant", "content": "Je vais bien, merci ! Et toi ?"}]}
{"messages": [{"role": "user", "content": "Explique-moi le machine learning"}, {"role": "assistant", "content": "Le machine learning est une branche de l'IA..."}]}
```

### Exemple de dataset

Voir [`examples/example_dataset.jsonl`](examples/example_dataset.jsonl) pour un exemple de 10 dialogues variés.

> **Recommandation** : 100-200+ dialogues de qualité pour de bons résultats

## 🧠 Pourquoi Vigostral-7B-Chat ?

| Aspect | Vigostral-7B-Chat | GPT-2 Français |
|--------|-------------------|----------------|
| **Taille** | 7 milliards de params | 1.7 milliards |
| **Entraînement** | 213k dialogues français | Texte générique |
| **Capacité conversationnelle** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐ Limité |
| **Architecture** | Mistral-7B (SOTA) | GPT-2 (2019) |
| **Multi-turn chat** | ✅ Oui | ❌ Non natif |

**Vigostral est spécifiquement entraîné pour tenir des conversations naturelles en français.**

## ⚙️ Technique : LoRA (Low-Rank Adaptation)

**Problème** : Fine-tuner 7 milliards de paramètres nécessite ~50GB de VRAM (impossible sur T4).

**Solution** : **LoRA** ne fine-tune que ~1% des paramètres (~70M) en ajoutant de petites matrices entraînables.

**Avantages** :
- ✅ Tient dans 16GB (GPU T4 gratuit)
- ✅ Entraînement 3-5× plus rapide
- ✅ Qualité comparable au fine-tuning complet
- ✅ Fichiers d'adaptateurs légers (~100-200 MB)

## 📁 Structure du projet

```
nanochat-french-tutorial/
├── vigostral_finetune_colab.ipynb  # Notebook Colab (fine-tuning)
├── chat_gradio.py                  # Interface web locale
├── INTEGRATION_NANOCHAT.md         # Guide d'intégration avec nanochat
├── README.md                        # Ce fichier
├── examples/
│   └── example_dataset.jsonl       # Exemple de dataset
└── .gitignore
```

## 📊 Performances attendues

### Sur GPU T4 (Google Colab gratuit)

| Métrique | Valeur |
|----------|--------|
| **Temps d'installation** | 2-3 minutes |
| **Temps de fine-tuning** (123 dialogues, 3 epochs) | 20-30 minutes |
| **Taille modèle base** | ~4 GB (quantization 4-bit) |
| **Taille adaptateurs LoRA** | ~100-200 MB |
| **Coût** | 0€ (GPU gratuit) |

### Comparaison CPU vs GPU

- **CPU (MacBook Pro)** : ❌ Impraticable (plusieurs heures bloqué)
- **GPU T4 (Colab)** : ✅ 20-30 minutes
- **Speedup** : **~100-200× plus rapide** ! 🚀

## 🛠️ Technologies utilisées

- [Vigostral-7B-Chat](https://huggingface.co/vigostral/vigostral-7b-chat) - Modèle conversationnel français
- [PEFT (LoRA)](https://huggingface.co/docs/peft) - Fine-tuning efficace
- [Transformers](https://huggingface.co/docs/transformers) - Bibliothèque HuggingFace
- [Gradio](https://gradio.app/) - Interface web moderne
- [Google Colab](https://colab.research.google.com/) - GPU gratuit
- [PyTorch](https://pytorch.org/) - Framework deep learning

## 🎓 Ce que vous apprendrez

1. **Fine-tuning avec LoRA** : Adapter un grand modèle avec peu de ressources
2. **Modèles conversationnels** : Différence entre GPT-2 et modèles de chat
3. **Quantization 4-bit** : Réduire la mémoire sans perdre de qualité
4. **Google Colab** : Utiliser des GPUs gratuits efficacement
5. **Gradio** : Créer des interfaces web pour vos modèles
6. **HuggingFace Transformers** : Charger et utiliser des modèles open source

## 🆘 Problèmes courants

### Le GPU n'est pas activé sur Colab
**Solution** : Runtime → Change runtime type → Sélectionnez **T4 GPU**

### "Out of Memory" pendant le training
**Solution** :
- Réduisez `per_device_train_batch_size` de 1 à ... attendez, c'est déjà à 1 !
- Augmentez `gradient_accumulation_steps` de 4 à 8
- Réduisez `max_seq_length` de 512 à 256

### Le modèle répond toujours la même chose
**Solutions** :
- Augmentez la **temperature** (de 0.7 à 0.9)
- Ajoutez plus de dialogues variés dans votre dataset
- Augmentez le nombre d'epochs (de 3 à 5)

### Le téléchargement du modèle échoue
**Solution** :
- Créez un token HuggingFace (gratuit) sur https://huggingface.co/settings/tokens
- Acceptez les conditions d'utilisation de Vigostral sur HuggingFace

### L'interface Gradio ne se lance pas localement
**Solution** :
- Vérifiez que les adaptateurs LoRA sont dans `./vigostral-finetuned-final/`
- Si vous n'avez pas de GPU local, le modèle tournera sur CPU (lent mais fonctionnel)

## 🔗 Intégration avec Nanochat

Vous voulez utiliser votre modèle avec l'interface complète de **Nanochat** (Andrej Karpathy) ?

👉 Consultez le guide détaillé : [`INTEGRATION_NANOCHAT.md`](INTEGRATION_NANOCHAT.md)

**TL;DR** : Nanochat utilise une architecture GPT custom incompatible avec Vigostral (Mistral). Deux options :
- **Option A** : Créer un wrapper pour charger des modèles HuggingFace dans nanochat
- **Option C** : Utiliser l'interface Gradio standalone fournie (recommandé pour commencer)

## 📚 Ressources complémentaires

- [Documentation Vigostral](https://huggingface.co/vigostral/vigostral-7b-chat)
- [Tutoriel LoRA officiel](https://huggingface.co/blog/lora)
- [Documentation PEFT](https://huggingface.co/docs/peft)
- [Documentation Gradio](https://gradio.app/docs/)
- [Nanochat (Andrej Karpathy)](https://github.com/karpathy/nanochat)

## 🤝 Contribuer

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez ce repo
2. Créez une branche (`git checkout -b feature/amélioration`)
3. Committez vos changements (`git commit -m 'Ajout d'une fonctionnalité'`)
4. Pushez (`git push origin feature/amélioration`)
5. Ouvrez une Pull Request

## ❓ FAQ

### Q : Combien de dialogues faut-il pour de bons résultats ?
**R** : Minimum 50, idéal 100-200+. La qualité est plus importante que la quantité.

### Q : Puis-je utiliser ce modèle commercialement ?
**R** : Vigostral-7B-Chat est sous licence Apache 2.0 (usage commercial autorisé). Vérifiez toujours les licences.

### Q : Le modèle fonctionne-t-il hors ligne ?
**R** : Oui ! Une fois téléchargé, tout fonctionne localement sans internet.

### Q : Quelle est la différence entre ce projet et nanochat ?
**R** :
- **Nanochat** : Framework complet pour entraîner un LLM from scratch (coûte $100, 4h sur 8×H100)
- **Ce projet** : Fine-tune un modèle français existant sur vos dialogues (gratuit, 30 min sur T4)

### Q : Puis-je fine-tuner sur d'autres langues ?
**R** : Oui ! Remplacez Vigostral par un autre modèle (ex: [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) pour l'anglais).

## ✨ Exemples de résultats

Après fine-tuning sur 123 dialogues techniques :

**Prompt** : *"Explique-moi ton projet principal"*

**Avant fine-tuning (modèle de base)** :
```
Mon projet principal consiste à développer des solutions d'IA...
[réponse générique]
```

**Après fine-tuning (votre style)** :
```
Mon projet principal c'est Geofen, une plateforme de jeu en ligne
où les joueurs peuvent créer leurs propres mécaniques de jeu avec
un système de règles modulaires. J'utilise Python et Flask pour
le backend, et j'expérimente avec des IA pour générer du contenu...
[réponse personnalisée avec votre vocabulaire et vos projets]
```

---

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

Le modèle [Vigostral-7B-Chat](https://huggingface.co/vigostral/vigostral-7b-chat) est sous licence Apache 2.0.

## 🙏 Remerciements

- [Vigogne AI](https://github.com/bofenghuang/vigogne) pour Vigostral-7B-Chat
- [Mistral AI](https://mistral.ai/) pour l'architecture Mistral-7B
- [HuggingFace](https://huggingface.co/) pour les outils et l'infrastructure
- [Andrej Karpathy](https://github.com/karpathy) pour l'inspiration (nanochat)
- La communauté IA française 🇫🇷

---

**Made with ❤️ for the French AI community**

Pour toute question, ouvrez une [issue](https://github.com/cladjidane/nanochat-french-tutorial/issues) !
