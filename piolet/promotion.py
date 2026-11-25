"""Garde de promotion : un nouveau modèle ne remplace l'ancien que s'il fait aussi bien partout.

Les deux modèles passent le **même** examen certificatif. Le candidat est promu si :

1. il acquiert toutes les compétences du référentiel ;
2. il ne recule sur aucune compétence (à la tolérance près) par rapport au modèle en place.

Une moyenne meilleure ne suffit pas : un recul sur une seule compétence bloque la promotion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .examen import Resultat


@dataclass(frozen=True)
class DecisionPromotion:
    promouvoir: bool
    raisons: list[str] = field(default_factory=list)


def decider(candidat: Resultat, en_place: Resultat | None = None, tolerance: float = 0.0) -> DecisionPromotion:
    raisons: list[str] = []
    manquantes = candidat.non_acquises()
    if manquantes:
        raisons.append("compétences non acquises : " + ", ".join(manquantes))
    if en_place is not None:
        ids_candidat = {n.competence for n in candidat.notes}
        for ancienne in en_place.notes:
            if ancienne.competence not in ids_candidat:
                raisons.append(f"{ancienne.competence} : absente de l'examen du candidat")
                continue
            nouvelle = candidat.note(ancienne.competence)
            if nouvelle.score + tolerance < ancienne.score:
                raisons.append(f"{ancienne.competence} : recul de {ancienne.score:.3f} à {nouvelle.score:.3f}")
    if not raisons:
        raisons.append("toutes les compétences acquises, aucun recul")
        return DecisionPromotion(True, raisons)
    return DecisionPromotion(False, raisons)
