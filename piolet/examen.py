"""Examen : un score par compétence, comparé à son seuil.

L'examen ne donne pas une moyenne : il donne une note par compétence, comme un relevé
de notes. Une compétence est acquise si sa note atteint son seuil. Le modèle n'est
certifiable que si **toutes** les compétences sont acquises.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .referentiel import Referentiel

_TOLERANCE = 1e-12


class ExamenIncomplet(RuntimeError):
    """Des items d'examen sont encore à rédiger : on ne fait pas passer un examen à trous."""


@dataclass(frozen=True)
class Note:
    competence: str
    bloc: str
    score: float
    seuil: float
    n: int

    def acquise(self, marge: float = 0.0) -> bool:
        return self.score + _TOLERANCE >= self.seuil + marge

    def en_dict(self) -> dict[str, Any]:
        return {
            "competence": self.competence,
            "bloc": self.bloc,
            "score": round(self.score, 4),
            "seuil": self.seuil,
            "n": self.n,
            "acquise": self.acquise(),
        }


@dataclass(frozen=True)
class Resultat:
    notes: tuple[Note, ...]

    def tout_acquis(self, marge: float = 0.0) -> bool:
        return all(n.acquise(marge) for n in self.notes)

    def non_acquises(self, marge: float = 0.0) -> list[str]:
        return [n.competence for n in self.notes if not n.acquise(marge)]

    def score_min(self) -> float:
        return min(n.score for n in self.notes)

    def score_moyen(self) -> float:
        return float(np.mean([n.score for n in self.notes]))

    def note(self, competence: str) -> Note:
        for n in self.notes:
            if n.competence == competence:
                return n
        raise KeyError(competence)

    def par_bloc(self) -> dict[str, list[Note]]:
        blocs: dict[str, list[Note]] = {}
        for n in self.notes:
            blocs.setdefault(n.bloc, []).append(n)
        return blocs

    def en_dict(self) -> dict[str, Any]:
        return {
            "tout_acquis": self.tout_acquis(),
            "score_min": round(self.score_min(), 4),
            "score_moyen": round(self.score_moyen(), 4),
            "notes": [n.en_dict() for n in self.notes],
        }


def examiner_classification(ref: Referentiel, y_vrai: Iterable[Any], y_pred: Iterable[Any]) -> Resultat:
    """F1 de chaque compétence (une étiquette chacune) sur les items de l'examen."""
    if ref.tache != "classification":
        raise ValueError("ce référentiel n'est pas un référentiel de classification")
    y_vrai = np.asarray(list(y_vrai), dtype=object)
    y_pred = np.asarray(list(y_pred), dtype=object)
    if y_vrai.shape != y_pred.shape:
        raise ValueError("y_vrai et y_pred doivent avoir la même longueur")
    notes = []
    for c in ref.competences:
        vrai = y_vrai == c.etiquette
        pred = y_pred == c.etiquette
        vp = int(np.sum(vrai & pred))
        fp = int(np.sum(~vrai & pred))
        fn = int(np.sum(vrai & ~pred))
        denominateur = 2 * vp + fp + fn
        f1 = 2 * vp / denominateur if denominateur else 0.0
        notes.append(Note(c.id, c.bloc, float(f1), c.seuil, int(np.sum(vrai))))
    return Resultat(tuple(notes))


def _normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKC", texte).casefold()
    return re.sub(r"\s+", " ", texte).strip()


def verifier_reponse(reponse: str, attendu: str, verification: str = "contient") -> bool:
    """Vérifie une réponse générée : égalité, inclusion ou expression régulière."""
    if verification == "egal":
        return _normaliser(reponse) == _normaliser(attendu)
    if verification == "contient":
        return _normaliser(attendu) in _normaliser(reponse)
    if verification == "regex":
        return re.search(attendu, reponse, flags=re.IGNORECASE | re.DOTALL) is not None
    raise ValueError(f"mode de vérification inconnu : {verification}")


def examiner_generation(ref: Referentiel, generer: Callable[[str], str]) -> Resultat:
    """Taux d'items réussis par compétence ; ``generer`` reçoit une consigne et rend une réponse."""
    if ref.tache != "generation":
        raise ValueError("ce référentiel n'est pas un référentiel de génération")
    manquants = ref.a_rediger()
    if manquants:
        raise ExamenIncomplet(f"{len(manquants)} item(s) d'examen à rédiger : {', '.join(manquants[:5])}…")
    notes = []
    for c in ref.competences:
        reussis = sum(verifier_reponse(generer(i.consigne), i.attendu, i.verification) for i in c.items)
        notes.append(Note(c.id, c.bloc, reussis / len(c.items), c.seuil, len(c.items)))
    return Resultat(tuple(notes))
