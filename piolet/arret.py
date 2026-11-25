"""Arrêt sur preuve : contrôle continu, puis examen de certification.

Un organisme de formation ne certifie pas un stagiaire sur ses exercices d'entraînement :
le contrôle continu guide la formation, un examen distinct décide du titre. Piolet
applique la même règle à un modèle, parce que s'arrêter au premier examen réussi est
trop optimiste : à force de réexaminer, on finit par tomber sur un examen « favorable ».

* L'**examen formatif** est passé à chaque tour. Il guide l'entraînement.
* Quand il dépasse chaque seuil d'une **marge** de sécurité, une **session** de
  certification s'ouvre : l'**examen certificatif**, scellé jusque-là, est passé une fois.
* Réussi : le modèle est certifié et l'entraînement s'arrête.
* Échoué : l'entraînement continue. La session suivante attend un **délai** (en tours),
  qui **double** après chaque échec : on ne repasse l'examen qu'après s'être entraîné
  davantage. Le nombre de sessions est **limité**. Chaque session est inscrite au relevé.

La lecture répétée d'un même jeu de test biaise les résultats : voir Dwork et al.,
« The reusable holdout: Preserving validity in adaptive data analysis », Science, 2015.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .examen import Resultat
from .referentiel import Referentiel


@dataclass(frozen=True)
class Decision:
    arreter: bool
    certifie: bool
    raison: str
    session: int | None = None
    certificatif: Resultat | None = None


@dataclass
class ArretSurPreuve:
    marge: float = 0.03
    sessions_max: int = 5
    delai: int = 3
    croissance: float = 2.0
    certification: bool = True
    sessions: int = field(default=0, init=False)
    dernier_tour: int | None = field(default=None, init=False)
    attente: int = field(default=0, init=False)

    @classmethod
    def depuis(cls, ref: Referentiel, **remplacements) -> "ArretSurPreuve":
        c = ref.certification
        params = {"marge": c.marge, "sessions_max": c.sessions_max, "delai": c.delai, "croissance": c.croissance}
        params.update(remplacements)
        return cls(**params)

    @classmethod
    def naif(cls) -> "ArretSurPreuve":
        """Arrêt au premier examen formatif réussi, sans marge ni certification (pour comparaison)."""
        return cls(marge=0.0, certification=False)

    def evaluer(self, tour: int, formatif: Resultat, certifier: Callable[[], Resultat]) -> Decision:
        """Décide, après le tour ``tour``, s'il faut continuer, certifier ou arrêter."""
        if not formatif.tout_acquis(self.marge):
            return Decision(False, False, "formatif_insuffisant")
        if not self.certification:
            return Decision(True, True, "formatif_reussi")
        if self.dernier_tour is not None and tour - self.dernier_tour < self.attente:
            return Decision(False, False, "delai")
        self.sessions += 1
        self.dernier_tour = tour
        certificatif = certifier()
        if certificatif.tout_acquis():
            return Decision(True, True, "certifie", self.sessions, certificatif)
        if self.sessions >= self.sessions_max:
            return Decision(True, False, "sessions_epuisees", self.sessions, certificatif)
        # attente avant la session suivante : délai, puis délai × croissance, etc.
        self.attente = int(math.ceil(self.delai * self.croissance ** (self.sessions - 1)))
        return Decision(False, False, "session_echouee", self.sessions, certificatif)

    def session_finale(self, certifier: Callable[[], Resultat]) -> Decision:
        """Budget épuisé : une dernière session, comme l'examen de fin de formation.

        Elle n'a lieu que si la certification est active et qu'il reste une session.
        """
        if not self.certification or self.sessions >= self.sessions_max:
            return Decision(True, False, "budget_epuise")
        self.sessions += 1
        certificatif = certifier()
        if certificatif.tout_acquis():
            return Decision(True, True, "certifie_fin_de_budget", self.sessions, certificatif)
        return Decision(True, False, "budget_epuise", self.sessions, certificatif)
