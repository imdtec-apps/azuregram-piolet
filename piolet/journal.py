"""Journal en ajout seul, chaîné par empreintes SHA-256 (FIPS 180-4).

Chaque entrée contient l'empreinte de la précédente : modifier une entrée, ou en
supprimer une au milieu, est détecté par ``verifier()``.

Un chaînage seul ne voit ni la suppression des dernières entrées, ni la réécriture
complète du fichier avec des empreintes recalculées. On conserve donc une **ancre**
(nombre d'entrées et dernière empreinte) hors du journal, signée si possible par une
clé secrète (HMAC-SHA-256, RFC 2104) : ``verifier(ancre)`` détecte alors ces deux cas.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ORIGINE = "0" * 64


def _empreinte(entree: dict) -> str:
    contenu = {k: v for k, v in entree.items() if k != "empreinte"}
    return hashlib.sha256(json.dumps(contenu, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _signature(entrees: int, empreinte: str, cle: bytes) -> str:
    return hmac.new(cle, f"{entrees}:{empreinte}".encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class Ancre:
    entrees: int
    empreinte: str
    signature: str = ""

    def signer(self, cle: bytes) -> "Ancre":
        return Ancre(self.entrees, self.empreinte, _signature(self.entrees, self.empreinte, cle))

    def signature_valide(self, cle: bytes) -> bool:
        return bool(self.signature) and hmac.compare_digest(self.signature, _signature(self.entrees, self.empreinte, cle))

    def enregistrer(self, chemin: str | Path) -> None:
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)
        Path(chemin).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def charger(cls, chemin: str | Path) -> "Ancre":
        return cls(**json.loads(Path(chemin).read_text(encoding="utf-8")))


class Journal:
    def __init__(self, chemin: str | Path, horloge: Callable[[], str] | None = None):
        self.chemin = Path(chemin)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self.horloge = horloge or (lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def lire(self) -> list[dict]:
        if not self.chemin.exists():
            return []
        return [json.loads(l) for l in self.chemin.read_text(encoding="utf-8").splitlines() if l.strip()]

    def ajouter(self, evenement: str, acteur: str, donnees: dict[str, Any]) -> dict:
        entrees = self.lire()
        entree = {
            "index": len(entrees),
            "horodatage": self.horloge(),
            "evenement": evenement,
            "acteur": acteur,
            "donnees": donnees,
            "empreinte_precedente": entrees[-1]["empreinte"] if entrees else ORIGINE,
        }
        entree["empreinte"] = _empreinte(entree)
        with self.chemin.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entree, ensure_ascii=False) + "\n")
        return entree

    def ancre(self) -> Ancre:
        """État du journal à conserver ailleurs (autre serveur, document signé, registre public)."""
        entrees = self.lire()
        return Ancre(len(entrees), entrees[-1]["empreinte"] if entrees else ORIGINE)

    def verifier(self, ancre: Ancre | None = None) -> tuple[bool, int | None]:
        """(True, None) si le journal est intact, sinon (False, index de la première anomalie)."""
        entrees = self.lire()
        precedente = ORIGINE
        for i, entree in enumerate(entrees):
            if entree.get("index") != i or entree.get("empreinte_precedente") != precedente:
                return False, i
            if entree.get("empreinte") != _empreinte(entree):
                return False, i
            precedente = entree["empreinte"]
        if ancre is not None and ancre.entrees > 0:
            if len(entrees) < ancre.entrees:
                return False, len(entrees)  # dernières entrées supprimées
            if entrees[ancre.entrees - 1]["empreinte"] != ancre.empreinte:
                return False, ancre.entrees - 1  # journal réécrit
        return True, None
