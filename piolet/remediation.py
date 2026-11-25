"""Remédiation : n'entraîner que ce qui manque, et s'arrêter dès la preuve.

Deux boucles :

* ``remedier`` : on achète des étiquettes par lots. À chaque tour, le modèle est
  réentraîné puis passe l'examen formatif. S'il échoue, on fait étiqueter les cas
  du réservoir où il hésite le plus (stratégie ``incertitude``, Settles, 2009).
* ``entrainer_par_epoques`` : toutes les données sont étiquetées ; on compte les
  époques. L'examen formatif est passé après chaque époque.

Dans les deux cas, ``ArretSurPreuve`` décide quand ouvrir une session de certification
et quand arrêter. ``proposer_lot`` sert aussi seul, pour une boucle avec de vrais
formateurs : il rend les indices des cas à leur soumettre.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .arret import ArretSurPreuve, Decision
from .examen import Resultat, examiner_classification
from .referentiel import Referentiel

STRATEGIES = ("incertitude", "uniforme", "ciblee")


def marges(proba: np.ndarray) -> np.ndarray:
    """Écart entre les deux classes les plus probables : petit écart, forte hésitation."""
    if proba.shape[1] < 2:
        return np.ones(len(proba))
    tri = np.sort(proba, axis=1)
    return tri[:, -1] - tri[:, -2]


def priorites(
    strategie: str,
    proba: np.ndarray,
    classes: np.ndarray,
    etiquettes_non_acquises: Sequence[Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Priorité de chaque cas du réservoir (plus grand = à étiqueter d'abord).

    * ``incertitude`` : 1 − marge.
    * ``uniforme`` : au hasard (référence).
    * ``ciblee`` : incertitude pondérée par la probabilité que le cas relève d'une
      compétence non acquise. Variante testée sur le banc : elle ne fait pas mieux que
      ``incertitude`` (voir banc/RESULTATS.md) ; elle reste disponible pour d'autres données.
    """
    if strategie not in STRATEGIES:
        raise ValueError(f"stratégie inconnue : {strategie} ({', '.join(STRATEGIES)})")
    bruit = 1e-9 * rng.random(len(proba))  # départage les égalités sans biais
    if strategie == "uniforme":
        return rng.random(len(proba))
    incertitude = 1.0 - marges(proba)
    if strategie == "incertitude":
        return incertitude + bruit
    colonnes = np.isin(classes, np.asarray(list(etiquettes_non_acquises), dtype=classes.dtype))
    poids = 0.05 + proba[:, colonnes].sum(axis=1)
    return incertitude * poids + bruit


def proposer_lot(
    modele: Any,
    X_reservoir: Any,
    lot: int,
    *,
    strategie: str = "incertitude",
    etiquettes_non_acquises: Sequence[Any] = (),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Indices (dans ``X_reservoir``) des ``lot`` cas à faire étiqueter en priorité."""
    rng = rng or np.random.default_rng(0)
    n = len(X_reservoir)
    if n == 0:
        return np.array([], dtype=int)
    if strategie == "uniforme":
        return rng.choice(n, size=min(lot, n), replace=False)
    p = priorites(strategie, modele.proba(X_reservoir), modele.classes, etiquettes_non_acquises, rng)
    return np.argsort(-p)[:lot]


@dataclass
class Tour:
    tour: int
    etiquettes: int
    formatif: Resultat
    decision: Decision

    def en_dict(self) -> dict[str, Any]:
        d = {
            "tour": self.tour,
            "etiquettes": self.etiquettes,
            "formatif_min": round(self.formatif.score_min(), 4),
            "decision": self.decision.raison,
        }
        if self.decision.session is not None:
            d["session"] = self.decision.session
            d["certificatif_min"] = round(self.decision.certificatif.score_min(), 4)
        return d


@dataclass
class Trajectoire:
    tours: list[Tour] = field(default_factory=list)
    certifie: bool = False
    raison: str = ""
    etiquettes: int = 0
    epoques: int = 0
    secondes: float = 0.0
    sessions: int = 0
    formatif: Resultat | None = None
    certificatif: Resultat | None = None

    def en_dict(self) -> dict[str, Any]:
        return {
            "certifie": self.certifie,
            "raison": self.raison,
            "etiquettes": self.etiquettes,
            "epoques": self.epoques,
            "secondes": round(self.secondes, 4),
            "sessions": self.sessions,
            "tours": [t.en_dict() for t in self.tours],
            "formatif": self.formatif.en_dict() if self.formatif else None,
            "certificatif": self.certificatif.en_dict() if self.certificatif else None,
        }


def _journaliser(journal: Any, evenement: str, donnees: dict[str, Any]) -> None:
    if journal is not None:
        journal.ajouter(evenement, "piolet", donnees)


def remedier(
    ref: Referentiel,
    modele: Any,
    X_reservoir: Any,
    etiqueter: Callable[[np.ndarray], Sequence[Any]],
    *,
    formatif: tuple[Any, Any],
    certificatif: tuple[Any, Any],
    amorce: Sequence[int],
    lot: int = 10,
    budget: int | None = None,
    strategie: str = "incertitude",
    arret: ArretSurPreuve | None = None,
    graine: int = 0,
    journal: Any = None,
) -> Trajectoire:
    """Achète des étiquettes par lots jusqu'à la certification (ou l'épuisement du budget).

    ``etiqueter(indices)`` rend les étiquettes des cas demandés : un formateur dans la
    réalité, un tableau connu dans un banc d'essai. ``budget`` compte toutes les
    étiquettes, amorce comprise. Le temps mesuré est celui du calcul (entraînement et
    examens), pas celui de l'étiquetage.
    """
    arret = arret or ArretSurPreuve.depuis(ref)
    rng = np.random.default_rng(graine)
    if isinstance(X_reservoir, (list, tuple)):
        X_reservoir = np.asarray(X_reservoir, dtype=object)  # textes : indexation par tableau
    n = len(X_reservoir)
    budget = n if budget is None else min(budget, n)
    X_f, y_f = formatif
    X_c, y_c = certificatif
    etiquetes = [int(i) for i in amorce]
    y_etiquetes = list(etiqueter(np.asarray(etiquetes, dtype=int)))
    disponibles = np.ones(n, dtype=bool)
    disponibles[etiquetes] = False
    traj = Trajectoire()
    _journaliser(journal, "debut", {"referentiel": ref.empreinte(), "strategie": strategie, "lot": lot, "amorce": len(etiquetes)})
    tour = 0
    while True:
        t0 = time.perf_counter()
        X_lab = X_reservoir[etiquetes]
        modele.entrainer(X_lab, np.asarray(y_etiquetes))
        res_f = examiner_classification(ref, y_f, modele.predire(X_f))
        decision = arret.evaluer(tour, res_f, lambda: examiner_classification(ref, y_c, modele.predire(X_c)))
        traj.secondes += time.perf_counter() - t0
        traj.tours.append(Tour(tour, len(etiquetes), res_f, decision))
        _journaliser(journal, "examen_formatif", {"tour": tour, "etiquettes": len(etiquetes), "score_min": round(res_f.score_min(), 4), "non_acquises": res_f.non_acquises()})
        if decision.session is not None:
            _journaliser(journal, "session_certification", {"tour": tour, "session": decision.session, "reussie": decision.certifie, "score_min": round(decision.certificatif.score_min(), 4)})
        traj.formatif = res_f
        if decision.certificatif is not None:
            traj.certificatif = decision.certificatif
        if decision.arreter:
            traj.raison = decision.raison
            traj.certifie = decision.certifie
            break
        if len(etiquetes) >= budget or not disponibles.any():
            finale = arret.session_finale(lambda: examiner_classification(ref, y_c, modele.predire(X_c)))
            if finale.session is not None:
                traj.certificatif = finale.certificatif
                _journaliser(journal, "session_certification", {"tour": tour, "session": finale.session, "reussie": finale.certifie, "score_min": round(finale.certificatif.score_min(), 4), "finale": True})
            traj.raison, traj.certifie = finale.raison, finale.certifie
            break
        restants = np.flatnonzero(disponibles)
        non_acquises = ref.etiquettes(res_f.non_acquises(arret.marge))
        t0 = time.perf_counter()
        choix = proposer_lot(modele, X_reservoir[restants], min(lot, budget - len(etiquetes)), strategie=strategie, etiquettes_non_acquises=non_acquises, rng=rng)
        traj.secondes += time.perf_counter() - t0
        nouveaux = restants[choix]
        y_etiquetes.extend(etiqueter(nouveaux))
        etiquetes.extend(int(i) for i in nouveaux)
        disponibles[nouveaux] = False
        tour += 1
    traj.etiquettes = len(etiquetes)
    traj.sessions = arret.sessions
    _journaliser(journal, "fin", {"raison": traj.raison, "certifie": traj.certifie, "etiquettes": traj.etiquettes, "sessions": traj.sessions})
    return traj


def entrainer_par_epoques(
    ref: Referentiel,
    modele: Any,
    X: Any,
    y: Any,
    *,
    formatif: tuple[Any, Any],
    certificatif: tuple[Any, Any],
    epoques_max: int = 200,
    arret: ArretSurPreuve | None = None,
    journal: Any = None,
) -> Trajectoire:
    """Entraîne époque par époque et s'arrête dès la certification (ou à ``epoques_max``)."""
    arret = arret or ArretSurPreuve.depuis(ref)
    X_f, y_f = formatif
    X_c, y_c = certificatif
    traj = Trajectoire(etiquettes=len(y))
    _journaliser(journal, "debut", {"referentiel": ref.empreinte(), "epoques_max": epoques_max})
    for ep in range(1, epoques_max + 1):
        t0 = time.perf_counter()
        modele.epoque(X, y)
        res_f = examiner_classification(ref, y_f, modele.predire(X_f))
        decision = arret.evaluer(ep, res_f, lambda: examiner_classification(ref, y_c, modele.predire(X_c)))
        traj.secondes += time.perf_counter() - t0
        traj.tours.append(Tour(ep, len(y), res_f, decision))
        if decision.session is not None:
            _journaliser(journal, "session_certification", {"epoque": ep, "session": decision.session, "reussie": decision.certifie, "score_min": round(decision.certificatif.score_min(), 4)})
        traj.formatif = res_f
        if decision.certificatif is not None:
            traj.certificatif = decision.certificatif
        traj.epoques = ep
        if decision.arreter:
            traj.raison, traj.certifie = decision.raison, decision.certifie
            break
    else:
        finale = arret.session_finale(lambda: examiner_classification(ref, y_c, modele.predire(X_c)))
        if finale.session is not None:
            traj.certificatif = finale.certificatif
            _journaliser(journal, "session_certification", {"epoque": traj.epoques, "session": finale.session, "reussie": finale.certifie, "score_min": round(finale.certificatif.score_min(), 4), "finale": True})
        traj.raison, traj.certifie = finale.raison, finale.certifie
    traj.sessions = arret.sessions
    _journaliser(journal, "fin", {"raison": traj.raison, "certifie": traj.certifie, "epoques": traj.epoques, "sessions": traj.sessions})
    return traj
