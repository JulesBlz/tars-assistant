"""
Fusionne le LoRA v2 avec Llama 3.1 8B en local.
Sortie : ~/tars/finetune/tars_v2_merged/ (safetensors fusionnés)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE = "unsloth/Meta-Llama-3.1-8B-Instruct"
LORA = os.path.expanduser("~/tars/finetune/tars_lora_v2")
OUT = os.path.expanduser("~/tars/finetune/tars_v2_merged")

print("Chargement du modèle de base (téléchargement ~16 GB si absent, puis chargement)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print("Application du LoRA v2...")
model = PeftModel.from_pretrained(base_model, LORA)

print("Fusion des poids...")
model = model.merge_and_unload()

print("Sauvegarde du modèle fusionné...")
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.save_pretrained(OUT)

print(f"Fusion terminée. Modèle dans {OUT}")
