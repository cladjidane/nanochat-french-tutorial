# 🔧 Errata & Corrections

## ✅ Correction du nom du modèle Vigostral (2025-10-17)

### Problème

Dans la version initiale du projet, le nom du modèle HuggingFace était incorrect :
- ❌ **Incorrect** : `vigostral/vigostral-7b-chat`
- ✅ **Correct** : `bofenghuang/vigostral-7b-chat`

### Erreur rencontrée

Si vous utilisez l'ancien nom, vous obtiendrez cette erreur :

```
RepositoryNotFoundError: 404 Client Error.
Repository Not Found for url: https://huggingface.co/vigostral/vigostral-7b-chat/resolve/main/config.json.
```

### Fichiers corrigés

Les fichiers suivants ont été mis à jour avec le bon nom de modèle :

1. ✅ **`vigostral_finetune_colab.ipynb`**
   - Ligne : `model_name = "bofenghuang/vigostral-7b-chat"`
   - Cellule de test final également corrigée

2. ✅ **`chat_gradio.py`**
   - Ligne 20 : `MODEL_NAME = "bofenghuang/vigostral-7b-chat"`

3. ✅ **`INTEGRATION_NANOCHAT.md`**
   - Ajout d'une note explicative

### Solution si vous avez déjà commencé

Si vous avez déjà uploadé le notebook sur Colab avec l'ancien nom :

**Option 1** : Re-téléchargez le notebook depuis GitHub
1. Allez sur https://github.com/cladjidane/nanochat-french-tutorial
2. Téléchargez le nouveau `vigostral_finetune_colab.ipynb`
3. Uploadez-le à nouveau sur Colab

**Option 2** : Modifiez manuellement dans Colab
1. Dans Colab, trouvez la cellule avec `model_name = "vigostral/vigostral-7b-chat"`
2. Changez en : `model_name = "bofenghuang/vigostral-7b-chat"`
3. Ré-exécutez la cellule

### Informations sur le modèle

**Modèle Vigostral-7B-Chat**
- **Créateur** : Bo Feng Huang (bofenghuang)
- **Organisation** : Vigogne AI
- **URL HuggingFace** : https://huggingface.co/bofenghuang/vigostral-7b-chat
- **Licence** : Apache 2.0
- **Base** : Mistral-7B-v0.1
- **Données d'entraînement** : ~213k dialogues français (distillés de GPT-3.5/4)

### Formats disponibles

Le modèle Vigostral-7B-Chat est disponible en plusieurs formats :

- **Standard** : `bofenghuang/vigostral-7b-chat` (utilisé dans ce projet)
- **GGUF** : `FlorianJc/Vigostral-7b-Chat-GGUF` (pour llama.cpp)
- **GGUF** : `TheBloke/Vigostral-7B-Chat-GGUF` (alternative)
- **AWQ** : Versions quantizées pour inference rapide
- **GPTQ** : Versions quantizées alternatives

Pour ce projet, nous utilisons la **version standard** car elle fonctionne parfaitement avec Transformers + PEFT (LoRA).

---

**Date de correction** : 2025-10-17
**Signalé par** : fabiencanu (cladjidane)
