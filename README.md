# 🇫🇷 Fine-tuning GPT-2 Français avec Nanochat

Tutorial complet pour fine-tuner le modèle GPT-2 français (`asi/gpt-fr-cased-base`) sur vos propres données, directement dans Google Colab avec un GPU gratuit.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VOTRE-USERNAME/nanochat-french-tutorial/blob/main/nanochat_french_colab.ipynb)

## 🎯 Objectif

Ce notebook vous guide pas à pas pour :
- ✅ Fine-tuner GPT-2 français sur vos dialogues personnalisés
- ✅ Utiliser un GPU gratuit (Google Colab T4)
- ✅ Obtenir un modèle personnalisé en 15-20 minutes
- ✅ Télécharger votre modèle pour l'utiliser localement

## 📋 Prérequis

1. **Un compte Google** (pour accéder à Google Colab)
2. **Un dataset de dialogues** au format JSONL (voir section [Format du dataset](#format-du-dataset))
3. **15-20 minutes** de votre temps

Aucune installation locale requise ! Tout se fait dans le navigateur.

## 🚀 Démarrage rapide

### Option 1 : Utiliser le notebook Colab (Recommandé)

1. Cliquez sur le badge "Open in Colab" ci-dessus
2. **Runtime** → **Change runtime type** → Sélectionnez **T4 GPU**
3. Suivez les instructions cellule par cellule
4. Uploadez votre dataset quand demandé
5. Lancez le training !

### Option 2 : Cloner et adapter localement

```bash
git clone https://github.com/VOTRE-USERNAME/nanochat-french-tutorial.git
cd nanochat-french-tutorial

# Ouvrir le notebook dans Jupyter/VSCode
jupyter notebook nanochat_french_colab.ipynb
```

## 📊 Format du dataset

Votre fichier `combined_dataset.jsonl` doit contenir des dialogues au format suivant :

```jsonl
{"messages": [{"role": "user", "content": "Bonjour, comment allez-vous ?"}, {"role": "assistant", "content": "Je vais bien, merci ! Et vous ?"}]}
{"messages": [{"role": "user", "content": "Expliquez-moi le machine learning"}, {"role": "assistant", "content": "Le machine learning est une branche de l'IA qui permet aux ordinateurs d'apprendre..."}]}
```

### Structure

- **Fichier** : Format JSONL (une ligne = un dialogue)
- **Clé `messages`** : Liste de messages alternant `user` et `assistant`
- **Encodage** : UTF-8

### Exemple de dataset minimal

Créez un fichier `example_dataset.jsonl` :

```jsonl
{"messages": [{"role": "user", "content": "Qui est Victor Hugo ?"}, {"role": "assistant", "content": "Victor Hugo (1802-1885) est un écrivain français majeur du XIXe siècle, auteur des Misérables et de Notre-Dame de Paris."}]}
{"messages": [{"role": "user", "content": "Quelle est la capitale de la France ?"}, {"role": "assistant", "content": "La capitale de la France est Paris."}]}
{"messages": [{"role": "user", "content": "Comment faire une omelette ?"}, {"role": "assistant", "content": "Pour faire une omelette : battez 2-3 œufs, ajoutez sel et poivre, versez dans une poêle chaude avec du beurre, laissez cuire 2-3 minutes."}]}
```

> **Note** : Pour de bons résultats, visez **200+ dialogues** de qualité.

## 📁 Structure du projet

```
nanochat-french-tutorial/
├── nanochat_french_colab.ipynb  # Le notebook principal
├── README.md                     # Ce fichier
├── examples/                     # Exemples de datasets
│   └── example_dataset.jsonl
└── .gitignore
```

## 🎓 Ce que vous apprendrez

En suivant ce tutorial, vous découvrirez :

1. **Fine-tuning de LLMs** : Comment adapter un modèle pré-entraîné à votre cas d'usage
2. **GPT-2 Architecture** : Comprendre le modèle GPT-2 français
3. **Google Colab** : Utiliser des GPUs gratuits pour l'entraînement
4. **Dataset preparation** : Formater vos données pour l'entraînement
5. **Évaluation** : Tester et valider votre modèle fine-tuné
6. **Déploiement** : Télécharger et utiliser votre modèle localement

## ⚙️ Paramètres d'entraînement

Les paramètres par défaut sont optimisés pour un bon compromis qualité/temps :

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `NUM_EPOCHS` | 3 | Nombre de passages sur le dataset |
| `DEVICE_BATCH_SIZE` | 32 | Nombre d'exemples par batch |
| `EMBEDDING_LR` | 0.2 | Learning rate pour les embeddings |
| `MATRIX_LR` | 0.02 | Learning rate pour les matrices |

Ces paramètres peuvent être modifiés dans la cellule "Configuration" du notebook.

## 📊 Performances attendues

Sur un GPU T4 gratuit (Google Colab) :

- **Temps d'installation** : 2-3 minutes
- **Temps de training** (300 iterations) : 10-15 minutes
- **Taille du modèle** : ~1.7 GB
- **Coût** : 0€ (GPU gratuit)

Comparé à un CPU (MacBook Pro) :
- Training sur CPU : **60+ heures** ❌
- Training sur T4 GPU : **10 minutes** ✅

C'est **360× plus rapide** ! 🚀

## 🛠️ Technologies utilisées

- [Nanochat](https://github.com/keller-jordan/nanochat) - Framework de training LLM minimaliste
- [GPT-2 French](https://huggingface.co/asi/gpt-fr-cased-base) - Modèle de base pré-entraîné
- [Google Colab](https://colab.research.google.com/) - Environnement GPU gratuit
- [PyTorch](https://pytorch.org/) - Framework deep learning
- [Transformers](https://huggingface.co/transformers/) - Bibliothèque Hugging Face

## 🆘 Problèmes courants

### Le GPU n'est pas activé
**Solution** : Runtime → Change runtime type → Sélectionnez T4 GPU

### "Out of Memory" pendant le training
**Solution** : Réduisez `DEVICE_BATCH_SIZE` de 32 à 16 ou 8

### Le modèle génère du charabia
**Solutions** :
- Augmentez le nombre d'époques (`NUM_EPOCHS = 5`)
- Vérifiez la qualité de votre dataset
- Ajoutez plus d'exemples (200+ recommandé)

### Erreur lors de l'upload du dataset
**Solution** : Vérifiez que votre fichier est bien au format JSONL UTF-8

## 🤝 Contribuer

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez ce repo
2. Créez une branche (`git checkout -b feature/amélioration`)
3. Committez vos changements (`git commit -m 'Ajout d'une fonctionnalité'`)
4. Pushez (`git push origin feature/amélioration`)
5. Ouvrez une Pull Request

## 📚 Ressources complémentaires

- [Documentation Nanochat](https://github.com/keller-jordan/nanochat)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Google Colab Tips](https://colab.research.google.com/notebooks/welcome.ipynb)

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

Le projet [Nanochat](https://github.com/keller-jordan/nanochat) original est également sous licence MIT.

## 🙏 Remerciements

- [keller-jordan](https://github.com/keller-jordan) pour le framework Nanochat
- [asi](https://huggingface.co/asi) pour le modèle GPT-2 français pré-entraîné
- La communauté Hugging Face pour les outils et modèles

## ✨ Exemples de résultats

Après fine-tuning sur un dataset de 300 dialogues techniques :

**Avant (modèle de base)** :
```
Prompt: "Expliquez-moi le machine learning"
Sortie: "Le machine learning est un domaine de recherche..."
```

**Après (fine-tuné)** :
```
Prompt: "Expliquez-moi le machine learning"
Sortie: "Le machine learning est une branche de l'intelligence artificielle
qui permet aux ordinateurs d'apprendre à partir de données sans être
explicitement programmés. Il repose sur des algorithmes qui identifient
des patterns dans les données et les utilisent pour faire des prédictions..."
```

---

**Made with ❤️ for the French AI community**

Pour toute question, ouvrez une [issue](https://github.com/VOTRE-USERNAME/nanochat-french-tutorial/issues) !
