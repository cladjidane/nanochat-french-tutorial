"""
Interface de chat Gradio pour Vigostral-7B-Chat fine-tuné.

Installation :
    pip install transformers peft accelerate bitsandbytes gradio torch

Usage :
    python chat_gradio.py

    Puis ouvrez votre navigateur sur l'URL affichée (généralement http://127.0.0.1:7860)
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

# Configuration
MODEL_NAME = "bofenghuang/vigostral-7b-chat"
LORA_ADAPTER_PATH = "./vigostral-finetuned-final"  # Chemin vers vos adaptateurs LoRA

# Détection du device
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"✅ GPU détecté : {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("✅ Apple Silicon (MPS) détecté")
else:
    DEVICE = "cpu"
    print("⚠️  Aucun GPU détecté, utilisation du CPU (sera lent)")


def load_model():
    """
    Charge le modèle Vigostral-7B-Chat avec les adaptateurs LoRA fine-tunés.
    """
    print(f"\n📥 Chargement du modèle {MODEL_NAME}...")

    # Configuration de quantization 4-bit pour économiser la mémoire
    if DEVICE == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # CPU/MPS : pas de quantization 4-bit (pas supporté)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
            device_map=DEVICE,
            trust_remote_code=True,
        )

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Charger les adaptateurs LoRA si disponibles
    if os.path.exists(LORA_ADAPTER_PATH):
        print(f"📥 Chargement des adaptateurs LoRA depuis {LORA_ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        print("✅ Adaptateurs LoRA chargés (modèle fine-tuné)")
    else:
        print(f"⚠️  Adaptateurs LoRA non trouvés dans {LORA_ADAPTER_PATH}")
        print("   → Utilisation du modèle de base (non fine-tuné)")

    print(f"✅ Modèle chargé avec succès !")
    print(f"   Taille en mémoire : {model.get_memory_footprint() / 1e9:.2f} GB\n")

    return model, tokenizer


# Charger le modèle au démarrage
MODEL, TOKENIZER = load_model()


def generate_response(message, history, temperature=0.7, max_tokens=200, top_p=0.9):
    """
    Génère une réponse du modèle pour un message utilisateur.

    Args:
        message (str): Message de l'utilisateur
        history (list): Historique de la conversation (non utilisé pour l'instant)
        temperature (float): Contrôle la créativité (0.1 = très déterministe, 1.0 = très créatif)
        max_tokens (int): Nombre maximum de tokens à générer
        top_p (float): Nucleus sampling (0.9 recommandé)

    Returns:
        str: Réponse générée par le modèle
    """
    # Formater le prompt au format Vigostral (Llama-2 style)
    prompt = f"<s>[INST] {message} [/INST]"

    # Tokenizer
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)

    # Générer
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    # Décoder la réponse
    full_response = TOKENIZER.decode(outputs[0], skip_special_tokens=True)

    # Extraire seulement la réponse (après [/INST])
    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[1].strip()
    else:
        response = full_response

    return response


# Interface Gradio
with gr.Blocks(title="🇫🇷 Chat Français Personnalisé") as demo:
    gr.Markdown(
        """
        # 🇫🇷 Chat Français Personnalisé

        Propulsé par **Vigostral-7B-Chat** fine-tuné sur votre style personnel.

        ---
        """
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        height=500,
        show_label=True,
        avatar_images=(None, "🤖"),
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Votre message",
            placeholder="Tapez votre message ici...",
            show_label=False,
            scale=9,
        )
        send_btn = gr.Button("Envoyer", variant="primary", scale=1)

    with gr.Accordion("⚙️ Paramètres avancés", open=False):
        temperature = gr.Slider(
            minimum=0.1,
            maximum=1.5,
            value=0.7,
            step=0.1,
            label="Temperature (créativité)",
            info="Plus haut = plus créatif, plus bas = plus déterministe",
        )
        max_tokens = gr.Slider(
            minimum=50,
            maximum=500,
            value=200,
            step=10,
            label="Nombre maximum de tokens",
            info="Longueur maximale de la réponse",
        )
        top_p = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.05,
            label="Top-p (nucleus sampling)",
            info="Diversité des réponses",
        )

    clear_btn = gr.Button("🗑️ Effacer la conversation")

    def user_message(user_msg, history):
        """Ajoute le message utilisateur à l'historique."""
        return "", history + [[user_msg, None]]

    def bot_response(history, temp, max_tok, top_p_val):
        """Génère et ajoute la réponse du bot à l'historique."""
        user_msg = history[-1][0]
        bot_msg = generate_response(
            user_msg,
            history[:-1],
            temperature=temp,
            max_tokens=max_tok,
            top_p=top_p_val,
        )
        history[-1][1] = bot_msg
        return history

    # Événements
    msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot, temperature, max_tokens, top_p], chatbot
    )
    send_btn.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot, temperature, max_tokens, top_p], chatbot
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)

    gr.Markdown(
        """
        ---

        ### 📝 Notes

        - **Temperature** : Contrôle la créativité du modèle (0.7 recommandé)
        - **Max tokens** : Longueur maximale de la réponse
        - **Top-p** : Contrôle la diversité (0.9 recommandé)

        ### 🛠️ Informations techniques

        - Modèle : Vigostral-7B-Chat (7 milliards de paramètres)
        - Architecture : Mistral-7B
        - Fine-tuning : LoRA (Low-Rank Adaptation)
        - Device : """ + DEVICE + """

        ---

        **Créé avec ❤️ pour la communauté IA française**
        """
    )

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 Lancement de l'interface Gradio...")
    print("=" * 80 + "\n")

    demo.launch(
        share=False,  # Mettez True pour générer un lien public temporaire
        server_name="0.0.0.0",  # Accessible depuis le réseau local
        server_port=7860,
    )
