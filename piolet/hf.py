"""Greffon Hugging Face : arrêt sur preuve dans ``Trainer`` et un adaptateur LoRA par bloc.

* ``PioletCallback`` se branche sur ``transformers.Trainer`` (``callbacks=[...]``). Après
  chaque époque, il fait passer l'examen formatif, ouvre si besoin une session de
  certification, et arrête l'entraînement dès que le modèle est certifié
  (``control.should_training_stop = True``).
* ``config_lora`` prépare un adaptateur LoRA (Hu et al., 2021) ; ``dossier_adaptateur``
  range un adaptateur par bloc de compétences. Quand le référentiel change un bloc, on
  ne réentraîne que l'adaptateur de ce bloc.

``transformers`` et ``peft`` sont optionnels (``pip install "piolet[hf]"``). Sans eux,
ce module s'importe quand même : la logique de l'arrêt se teste sans GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .arret import ArretSurPreuve, Decision
from .examen import Resultat

try:  # pragma: no cover - dépend de l'environnement
    from transformers import TrainerCallback as _Base

    HF_DISPONIBLE = True
except Exception:  # pragma: no cover
    class _Base:  # repli minimal : mêmes méthodes appelées par Trainer
        pass

    HF_DISPONIBLE = False


class PioletCallback(_Base):
    def __init__(
        self,
        arret: ArretSurPreuve,
        examiner_formatif: Callable[[Any], Resultat],
        examiner_certificatif: Callable[[Any], Resultat],
        journal: Any = None,
    ):
        self.arret = arret
        self.examiner_formatif = examiner_formatif
        self.examiner_certificatif = examiner_certificatif
        self.journal = journal
        self.historique: list[dict[str, Any]] = []
        self.decision: Decision | None = None

    def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> Any:
        epoque = getattr(state, "epoch", None)
        tour = int(round(epoque)) if epoque is not None else len(self.historique) + 1
        formatif = self.examiner_formatif(model)
        decision = self.arret.evaluer(tour, formatif, lambda: self.examiner_certificatif(model))
        trace = {"epoque": tour, "formatif_min": round(formatif.score_min(), 4), "decision": decision.raison}
        if decision.session is not None:
            trace["session"] = decision.session
            trace["certificatif_min"] = round(decision.certificatif.score_min(), 4)
        self.historique.append(trace)
        if self.journal is not None:
            self.journal.ajouter("examen_formatif", "piolet", trace)
        if decision.arreter:
            self.decision = decision
            control.should_training_stop = True
        return control


def config_lora(
    rang: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    cibles: tuple[str, ...] = ("q_proj", "v_proj"),
    tache: str = "SEQ_CLS",
) -> Any:
    """``peft.LoraConfig`` pour un bloc. Les noms des ``cibles`` dépendent de l'architecture du modèle."""
    from peft import LoraConfig, TaskType

    return LoraConfig(r=rang, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(cibles), task_type=getattr(TaskType, tache))


def dossier_adaptateur(racine: str | Path, bloc: str) -> Path:
    """Un dossier par bloc : ``<racine>/adaptateurs/<bloc>/``."""
    return Path(racine) / "adaptateurs" / bloc


def parametres_entrainables(modele: Any) -> tuple[int, int]:
    """(paramètres entraînables, paramètres totaux) d'un modèle PyTorch."""
    entrainables = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modele.parameters())
    return entrainables, total
