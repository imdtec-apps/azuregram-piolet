"""Exemple : affiner un modèle de classification de textes par LoRA, avec arrêt sur preuve.

NON EXÉCUTÉ DANS L'INTÉGRATION CONTINUE. Il demande :
    pip install "piolet[hf]"
un modèle Hugging Face téléchargeable (à choisir, variable MODELE) et, de préférence, un GPU.
La logique de l'arrêt (PioletCallback) est testée sans GPU dans tests/test_boucles.py.

Données attendues (format de l'atelier, voir exemples/skillvora-dwwm/README.md) :
    etiquetes.csv, examen_formatif.csv, examen_certificatif.csv : id,texte,competence,provenance
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from piolet import ArretSurPreuve, Journal, Referentiel, examiner_classification, exiger
from piolet.atelier import lire_csv
from piolet.hf import PioletCallback, config_lora, dossier_adaptateur, parametres_entrainables

MODELE = "<nom du modèle de base sur le Hub>"  # à choisir ; les noms des cibles LoRA en dépendent
CIBLES = ("q_proj", "v_proj")
DOSSIER = Path(sys.argv[1] if len(sys.argv) > 1 else "exemples/skillvora-dwwm")
BLOC = "BC01"  # un adaptateur par bloc de compétences

ref = Referentiel.charger(DOSSIER / "referentiel.yaml")
competences = [c.etiquette for c in ref.bloc(BLOC).competences]
index = {c: i for i, c in enumerate(competences)}


def charger(nom):
    lignes = [l for l in lire_csv(DOSSIER / nom) if l["competence"] in index]
    exiger(lignes, ref.provenances_autorisees)  # aucune sortie de modèle tiers
    return [l["texte"] for l in lignes], [l["competence"] for l in lignes]


X, y = charger("etiquetes.csv")
Xf, yf = charger("examen_formatif.csv")
Xc, yc = charger("examen_certificatif.csv")

tok = AutoTokenizer.from_pretrained(MODELE)
base = AutoModelForSequenceClassification.from_pretrained(MODELE, num_labels=len(competences))
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    base.config.pad_token_id = tok.pad_token_id
modele = get_peft_model(base, config_lora(rang=8, cibles=CIBLES))
print("paramètres entraînables / totaux :", parametres_entrainables(modele))


class Jeu(torch.utils.data.Dataset):
    def __init__(self, textes, etiquettes):
        self.enc = tok(textes, truncation=True, max_length=96, padding="max_length")
        self.y = [index[e] for e in etiquettes]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        d["labels"] = torch.tensor(self.y[i])
        return d


@torch.no_grad()
def examiner(m, textes, etiquettes):
    m.eval()
    predictions = []
    for k in range(0, len(textes), 32):
        lot = tok(textes[k:k + 32], truncation=True, max_length=96, padding=True, return_tensors="pt").to(m.device)
        predictions += [competences[i] for i in m(**lot).logits.argmax(-1).tolist()]
    m.train()
    sous_ref = Referentiel.depuis_dict({**ref.brut, "blocs": [b for b in ref.brut["blocs"] if b["id"] == BLOC]})
    return examiner_classification(sous_ref, etiquettes, predictions)


journal = Journal(f"journal-{BLOC}.jsonl")
rappel = PioletCallback(ArretSurPreuve.depuis(ref, delai=2), lambda m: examiner(m, Xf, yf), lambda m: examiner(m, Xc, yc), journal=journal)
args = TrainingArguments(output_dir=f"sortie-{BLOC}", num_train_epochs=20, per_device_train_batch_size=16,
                         learning_rate=2e-4, save_strategy="no", report_to="none", logging_steps=20)
Trainer(model=modele, args=args, train_dataset=Jeu(X, y), callbacks=[rappel]).train()

print("décision :", rappel.decision)
if rappel.decision is not None and rappel.decision.certifie:
    modele.save_pretrained(dossier_adaptateur(".", BLOC))
    journal.ajouter("adaptateur", "piolet", {"bloc": BLOC, "dossier": str(dossier_adaptateur(".", BLOC))})
