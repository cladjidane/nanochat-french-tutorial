"""
Script de test pour Vigostral-7B-Chat (modèle de base, sans fine-tuning)

Usage:
    python test_vigostral_base.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 80)
print("🤖 Test de Vigostral-7B-Chat (modèle de base)")
print("=" * 80)

MODEL_NAME = "bofenghuang/vigostral-7b-chat"

# Détection du device
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"✅ GPU détecté : {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("✅ Apple Silicon (MPS) détecté")
else:
    DEVICE = "cpu"
    print("⚠️  CPU détecté (génération sera lente)")

print(f"\n📥 Chargement du modèle {MODEL_NAME}...")
print("⏳ Cela peut prendre quelques minutes...\n")

# Charger le modèle (sans quantization pour plus de simplicité)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE if DEVICE != "mps" else "cpu",  # MPS pas toujours stable
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"✅ Modèle chargé ! Taille en mémoire : {model.get_memory_footprint() / 1e9:.2f} GB\n")


def chat(prompt, max_new_tokens=150, temperature=0.7, top_p=0.9):
    """
    Génère une réponse du modèle.
    """
    # Formater le prompt au format Vigostral (Llama-2 style)
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"

    # Tokenizer
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Générer
    print("🤔 Génération en cours...", end=" ", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("✓")

    # Décoder
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraire seulement la réponse (après [/INST])
    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[1].strip()
    else:
        response = full_response

    return response


# Tests automatiques
print("=" * 80)
print("🧪 Tests automatiques")
print("=" * 80)

test_prompts = [
    "Bonjour, qui es-tu ?",
    "Explique-moi le machine learning en une phrase",
    "Quelle est la capitale de la France ?",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n[Test {i}/{len(test_prompts)}]")
    print(f"👤 User: {prompt}")
    response = chat(prompt, max_new_tokens=100)
    print(f"🤖 Assistant: {response}")
    print("-" * 80)

# Mode interactif
print("\n" + "=" * 80)
print("💬 Mode interactif")
print("=" * 80)
print("Tapez 'quit' ou 'exit' pour quitter\n")

while True:
    try:
        user_input = input("👤 Vous: ")
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Au revoir !")
        break

    if user_input.lower() in ['quit', 'exit', 'q']:
        print("👋 Au revoir !")
        break

    if not user_input.strip():
        continue

    response = chat(user_input)
    print(f"🤖 Assistant: {response}\n")
