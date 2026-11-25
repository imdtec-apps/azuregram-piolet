"""Garde de provenance : chaque exemple d'entraînement dit d'où il vient, sinon il est refusé.

Trois raisons de refuser les sorties d'un modèle tiers :

1. **le droit** : les conditions d'utilisation de nombreux fournisseurs de modèles
   interdisent d'utiliser leurs sorties pour entraîner un modèle concurrent ;
2. **la qualité** : entraîner un modèle sur des données générées par des modèles fait
   disparaître la diversité des données d'origine (Shumailov et al., Nature, 2024) ;
3. **la traçabilité** : un relevé de compétences n'a de valeur que si l'on sait qui a
   écrit les exemples.

La garde ne devine rien : elle lit le champ ``provenance`` et le compare à la liste
autorisée par le référentiel. Un exemple sans provenance est refusé.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

INTERDITES = frozenset({"api-tierce", "sortie-modele-tiers", "modele-tiers"})


class ProvenanceRefusee(ValueError):
    def __init__(self, bilan: "Bilan"):
        self.bilan = bilan
        apercu = ", ".join(f"{i} ({p or 'sans provenance'})" for i, p, _ in bilan.refuses[:5])
        super().__init__(f"{len(bilan.refuses)} exemple(s) refusé(s) par la garde de provenance : {apercu}")


@dataclass
class Bilan:
    acceptes: list[str] = field(default_factory=list)
    refuses: list[tuple[str, str, str]] = field(default_factory=list)
    par_provenance: Counter = field(default_factory=Counter)

    @property
    def conforme(self) -> bool:
        return not self.refuses

    def en_dict(self) -> dict[str, Any]:
        return {
            "acceptes": len(self.acceptes),
            "refuses": [{"id": i, "provenance": p, "raison": r} for i, p, r in self.refuses],
            "par_provenance": dict(self.par_provenance),
        }


def controler(exemples: Iterable[Mapping[str, Any]], autorisees: Iterable[str]) -> Bilan:
    autorisees = frozenset(autorisees) - INTERDITES  # une provenance interdite ne s'autorise pas
    bilan = Bilan()
    for k, ex in enumerate(exemples):
        ident = str(ex.get("id", k))
        prov = str(ex.get("provenance", "") or "").strip()
        bilan.par_provenance[prov or "sans provenance"] += 1
        if not prov:
            bilan.refuses.append((ident, prov, "provenance absente"))
        elif prov in INTERDITES:
            bilan.refuses.append((ident, prov, "sortie d'un modèle tiers"))
        elif prov not in autorisees:
            bilan.refuses.append((ident, prov, "provenance non autorisée par le référentiel"))
        else:
            bilan.acceptes.append(ident)
    return bilan


def exiger(exemples: Iterable[Mapping[str, Any]], autorisees: Iterable[str]) -> Bilan:
    """Comme ``controler``, mais lève ``ProvenanceRefusee`` au premier lot non conforme."""
    bilan = controler(list(exemples), autorisees)
    if not bilan.conforme:
        raise ProvenanceRefusee(bilan)
    return bilan
