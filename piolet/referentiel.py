"""Référentiel de compétences : ce que le modèle doit savoir faire, et à quel niveau.

Un référentiel Piolet se lit comme un titre professionnel : des **blocs**, des
**compétences**, et pour chacune un **seuil** à atteindre à l'examen. Il fixe aussi
les règles de certification (marge de sécurité, nombre de sessions, délai entre deux
sessions) et la liste des **provenances autorisées** pour les données d'entraînement.

Deux types de tâches :

* ``classification`` : chaque compétence correspond à une étiquette ; son score est le
  F1 de cette étiquette à l'examen ;
* ``generation`` : chaque compétence a ses items d'examen (consigne, réponse attendue,
  mode de vérification) ; son score est le taux d'items réussis.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

TACHES = ("classification", "generation")
VERIFICATIONS = ("egal", "contient", "regex")
_A_REDIGER = re.compile(r"\[\s*à\s+rédiger", re.IGNORECASE)


class ReferentielInvalide(ValueError):
    """Le référentiel ne respecte pas le schéma ; ``problemes`` les liste tous."""

    def __init__(self, problemes: list[str]):
        self.problemes = list(problemes)
        super().__init__("Référentiel invalide :\n- " + "\n- ".join(self.problemes))


def a_rediger(texte: Any) -> bool:
    """Vrai si le texte est un emplacement réservé, par exemple « [à rédiger par un formateur] »."""
    return isinstance(texte, str) and bool(_A_REDIGER.search(texte))


@dataclass(frozen=True)
class Item:
    id: str
    consigne: str
    attendu: str
    verification: str = "contient"
    provenance: str = "formateur"

    @property
    def a_rediger(self) -> bool:
        return a_rediger(self.consigne) or a_rediger(self.attendu)


@dataclass(frozen=True)
class Competence:
    id: str
    titre: str
    bloc: str
    seuil: float
    etiquette: Any = None
    items: tuple[Item, ...] = ()


@dataclass(frozen=True)
class Bloc:
    id: str
    titre: str
    competences: tuple[Competence, ...]


@dataclass(frozen=True)
class Certification:
    """Règles de l'arrêt sur preuve.

    * ``marge`` : l'examen formatif doit dépasser chaque seuil de cette marge avant
      qu'une session de certification soit ouverte ;
    * ``sessions_max`` : nombre maximal de sessions de certification ;
    * ``delai`` : nombre de tours (lot d'étiquettes ou époque) après une session échouée ;
    * ``croissance`` : facteur appliqué au délai après chaque nouvel échec (2 = il double).
    """

    marge: float = 0.03
    sessions_max: int = 5
    delai: int = 3
    croissance: float = 2.0


@dataclass(frozen=True)
class Referentiel:
    id: str
    titre: str
    tache: str
    blocs: tuple[Bloc, ...]
    certification: Certification = Certification()
    provenances_autorisees: frozenset[str] = frozenset({"formateur"})
    version: str = "1"
    source: str = ""
    brut: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    # ------------------------------------------------------------------ lecture
    @property
    def competences(self) -> list[Competence]:
        return [c for b in self.blocs for c in b.competences]

    def competence(self, cid: str) -> Competence:
        for c in self.competences:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def bloc(self, bid: str) -> Bloc:
        for b in self.blocs:
            if b.id == bid:
                return b
        raise KeyError(bid)

    def etiquettes(self, cids: list[str] | None = None) -> list[Any]:
        """Étiquettes des compétences données (toutes par défaut), pour une tâche de classification."""
        comps = self.competences if cids is None else [self.competence(c) for c in cids]
        return [c.etiquette for c in comps]

    def a_rediger(self) -> list[str]:
        """Identifiants des items d'examen encore à rédiger."""
        return [i.id for c in self.competences for i in c.items if i.a_rediger]

    def empreinte(self) -> str:
        """Empreinte SHA-256 du référentiel tel qu'il a été chargé : elle date et fige l'examen."""
        canon = json.dumps(self.brut, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def resume(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "titre": self.titre,
            "version": self.version,
            "source": self.source,
            "tache": self.tache,
            "empreinte": self.empreinte(),
            "blocs": [
                {
                    "id": b.id,
                    "titre": b.titre,
                    "competences": [{"id": c.id, "titre": c.titre, "seuil": c.seuil} for c in b.competences],
                }
                for b in self.blocs
            ],
            "certification": {
                "marge": self.certification.marge,
                "sessions_max": self.certification.sessions_max,
                "delai": self.certification.delai,
                "croissance": self.certification.croissance,
            },
            "provenances_autorisees": sorted(self.provenances_autorisees),
        }

    # --------------------------------------------------------------- chargement
    @classmethod
    def charger(cls, chemin: str | Path) -> "Referentiel":
        with open(chemin, encoding="utf-8") as f:
            donnees = yaml.safe_load(f)
        return cls.depuis_dict(donnees)

    @classmethod
    def depuis_dict(cls, d: Mapping[str, Any]) -> "Referentiel":
        problemes: list[str] = []
        if not isinstance(d, Mapping):
            raise ReferentielInvalide(["le fichier doit décrire un dictionnaire YAML"])

        for cle in ("id", "titre", "tache", "blocs"):
            if cle not in d:
                problemes.append(f"champ obligatoire manquant : {cle}")
        tache = d.get("tache")
        if tache is not None and tache not in TACHES:
            problemes.append(f"tache doit valoir {' ou '.join(TACHES)} (reçu : {tache!r})")

        seuil_defaut = d.get("seuil", 0.9)
        autorisees = frozenset(str(p) for p in d.get("provenances_autorisees", ["formateur"]))
        if not autorisees:
            problemes.append("provenances_autorisees ne peut pas être vide")

        cert_d = d.get("certification", {}) or {}
        certification = Certification(
            marge=float(cert_d.get("marge", Certification.marge)),
            sessions_max=int(cert_d.get("sessions_max", Certification.sessions_max)),
            delai=int(cert_d.get("delai", Certification.delai)),
            croissance=float(cert_d.get("croissance", Certification.croissance)),
        )
        if not 0 <= certification.marge < 0.5:
            problemes.append("certification.marge doit être comprise entre 0 et 0,5")
        if certification.sessions_max < 1:
            problemes.append("certification.sessions_max doit valoir au moins 1")
        if certification.delai < 0:
            problemes.append("certification.delai ne peut pas être négatif")
        if certification.croissance < 1:
            problemes.append("certification.croissance doit valoir au moins 1")

        ids_blocs: set[str] = set()
        ids_comp: set[str] = set()
        ids_items: set[str] = set()
        etiquettes_vues: set[str] = set()
        blocs: list[Bloc] = []
        for bd in d.get("blocs", []) or []:
            bid = str(bd.get("id", ""))
            if not bid:
                problemes.append("un bloc n'a pas d'id")
            elif bid in ids_blocs:
                problemes.append(f"id de bloc en double : {bid}")
            ids_blocs.add(bid)
            comps: list[Competence] = []
            for cd in bd.get("competences", []) or []:
                cid = str(cd.get("id", ""))
                if not cid:
                    problemes.append(f"une compétence du bloc {bid} n'a pas d'id")
                elif cid in ids_comp:
                    problemes.append(f"id de compétence en double : {cid}")
                ids_comp.add(cid)
                seuil = cd.get("seuil", seuil_defaut)
                try:
                    seuil = float(seuil)
                except (TypeError, ValueError):
                    problemes.append(f"{cid} : seuil non numérique ({seuil!r})")
                    seuil = 0.0
                if not 0 < seuil <= 1:
                    problemes.append(f"{cid} : le seuil doit être compris entre 0 (exclu) et 1")
                items: list[Item] = []
                if tache == "classification":
                    if "etiquette" not in cd:
                        problemes.append(f"{cid} : une compétence de classification doit avoir une etiquette")
                    else:
                        cle_etiquette = repr(cd["etiquette"])
                        if cle_etiquette in etiquettes_vues:
                            problemes.append(f"{cid} : étiquette déjà utilisée ({cd['etiquette']!r})")
                        etiquettes_vues.add(cle_etiquette)
                elif tache == "generation":
                    for idd in cd.get("items", []) or []:
                        iid = str(idd.get("id", ""))
                        if not iid:
                            problemes.append(f"{cid} : un item n'a pas d'id")
                        elif iid in ids_items:
                            problemes.append(f"id d'item en double : {iid}")
                        ids_items.add(iid)
                        item = Item(
                            id=iid,
                            consigne=str(idd.get("consigne", "")),
                            attendu=str(idd.get("attendu", "")),
                            verification=str(idd.get("verification", "contient")),
                            provenance=str(idd.get("provenance", "")),
                        )
                        if not item.consigne or not item.attendu:
                            problemes.append(f"{iid} : consigne et attendu sont obligatoires")
                        if item.verification not in VERIFICATIONS:
                            problemes.append(f"{iid} : verification doit valoir {', '.join(VERIFICATIONS)}")
                        elif item.verification == "regex" and not item.a_rediger:
                            try:
                                re.compile(item.attendu)
                            except re.error as e:
                                problemes.append(f"{iid} : expression régulière invalide ({e})")
                        if item.provenance not in autorisees:
                            problemes.append(f"{iid} : provenance {item.provenance!r} non autorisée")
                        items.append(item)
                    if not items:
                        problemes.append(f"{cid} : une compétence de génération doit avoir au moins un item")
                comps.append(
                    Competence(
                        id=cid,
                        titre=str(cd.get("titre", cid)),
                        bloc=bid,
                        seuil=seuil,
                        etiquette=cd.get("etiquette"),
                        items=tuple(items),
                    )
                )
            if not comps:
                problemes.append(f"le bloc {bid} n'a aucune compétence")
            blocs.append(Bloc(id=bid, titre=str(bd.get("titre", bid)), competences=tuple(comps)))
        if not blocs and "blocs" in d:
            problemes.append("le référentiel n'a aucun bloc")

        if problemes:
            raise ReferentielInvalide(problemes)
        return cls(
            id=str(d["id"]),
            titre=str(d["titre"]),
            tache=str(tache),
            blocs=tuple(blocs),
            certification=certification,
            provenances_autorisees=autorisees,
            version=str(d.get("version", "1")),
            source=str(d.get("source", "")),
            brut=json.loads(json.dumps(d, default=str)),
        )
