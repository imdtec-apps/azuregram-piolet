"""Atelier : la boucle Piolet avec de vrais formateurs, un tour à la fois.

Chaque appel à ``tour`` :

1. vérifie la provenance des exemples déjà étiquetés et des examens ;
2. entraîne le modèle, lui fait passer l'examen formatif, ouvre si besoin une session ;
3. écrit dans le journal ce qui s'est passé ;
4. si le modèle n'est pas certifié, écrit ``a_etiqueter.csv`` : les cas du réservoir où il
   hésite le plus, à faire étiqueter par les formateurs avant le tour suivant.

L'état (nombre de tours, sessions passées, attente) est relu dans le journal : le
journal **est** l'état. Formats CSV (UTF-8, séparateur virgule) :

* examens et exemples étiquetés : ``id,texte,competence,provenance`` ;
* réservoir : ``id,texte`` (sans étiquette).
"""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .arret import ArretSurPreuve
from .cout import ticket_entrainement
from .examen import ExamenIncomplet, examiner_classification
from .journal import Journal
from .modeles import charger_modele
from .provenance import exiger
from .referentiel import Referentiel, a_rediger
from .remediation import proposer_lot

COLONNES = ["id", "texte", "competence", "provenance"]


def lire_csv(chemin: str | Path) -> list[dict[str, str]]:
    with open(chemin, encoding="utf-8-sig", newline="") as f:
        return [{k: (v or "").strip() for k, v in ligne.items()} for ligne in csv.DictReader(f)]


def ecrire_csv(chemin: str | Path, lignes: list[dict[str, str]]) -> None:
    with open(chemin, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES)
        w.writeheader()
        w.writerows(lignes)


def examens(ref: Referentiel, dossier: Path) -> dict[str, Any]:
    """Charge les examens déclarés par le référentiel et compte les items par compétence."""
    decl = dict(ref.brut.get("examens") or {})
    minimum = int(decl.get("items_par_competence", 30))
    sortie: dict[str, Any] = {"minimum": minimum}
    for nom in ("formatif", "certificatif"):
        if nom not in decl:
            raise ExamenIncomplet(f"le référentiel ne déclare pas d'examen {nom}")
        lignes = [l for l in lire_csv(dossier / decl[nom]) if l.get("texte") and not a_rediger(l.get("texte"))]
        compte = {c.id: sum(1 for l in lignes if l.get("competence") == str(c.etiquette)) for c in ref.competences}
        sortie[nom] = {"lignes": lignes, "par_competence": compte, "complet": all(v >= minimum for v in compte.values())}
    return sortie


def _etat(journal: Journal, arret: ArretSurPreuve) -> tuple[int, bool, float]:
    """Relit le journal : (tours passés, déjà certifié ?, secondes de calcul cumulées)."""
    tours, certifie, secondes, derniere = 0, False, 0.0, None
    for e in journal.lire():
        if e["evenement"] == "examen_formatif":
            tours += 1
            secondes += float(e["donnees"].get("secondes", 0))
        elif e["evenement"] == "session_certification":
            arret.sessions += 1
            derniere = e["donnees"]["tour"]
        elif e["evenement"] == "fin" and e["donnees"].get("certifie"):
            certifie = True
    if arret.sessions:
        arret.dernier_tour = derniere
        arret.attente = int(math.ceil(arret.delai * arret.croissance ** (arret.sessions - 1)))
    return tours, certifie, secondes


def tour(
    referentiel: str | Path,
    etiquetes: str | Path,
    reservoir: str | Path,
    journal: str | Path,
    *,
    lot: int = 20,
    modele: str = "texte-logistique",
    sortie_lot: str | Path = "a_etiqueter.csv",
    sortie_releve: str | Path = "releve.json",
) -> dict[str, Any]:
    ref_chemin = Path(referentiel)
    ref = Referentiel.charger(ref_chemin)
    ex = examens(ref, ref_chemin.parent)
    if not (ex["formatif"]["complet"] and ex["certificatif"]["complet"]):
        manque = {n: {k: v for k, v in ex[n]["par_competence"].items() if v < ex["minimum"]} for n in ("formatif", "certificatif")}
        raise ExamenIncomplet(f"examens incomplets (minimum {ex['minimum']} items par compétence) : {json.dumps(manque, ensure_ascii=False)}")
    exiger(ex["formatif"]["lignes"] + ex["certificatif"]["lignes"], ref.provenances_autorisees)
    lab = lire_csv(etiquetes)
    exiger(lab, ref.provenances_autorisees)
    valides = {str(c.etiquette) for c in ref.competences}
    inconnues = sorted({l["competence"] for l in lab} - valides)
    if inconnues:
        raise ValueError(f"compétences inconnues dans {etiquetes} : {', '.join(inconnues)}")

    j = Journal(journal)
    arret = ArretSurPreuve.depuis(ref)
    n_tours, deja, secondes_avant = _etat(j, arret)
    if deja:
        return {"certifie": True, "message": "modèle déjà certifié : voir le journal"}
    if n_tours == 0:
        j.ajouter("referentiel", "piolet", {"id": ref.id, "version": ref.version, "empreinte": ref.empreinte()})

    t0 = time.perf_counter()
    m = charger_modele(modele)
    m.entrainer([l["texte"] for l in lab], [l["competence"] for l in lab])
    Xf = [l["texte"] for l in ex["formatif"]["lignes"]]
    yf = [l["competence"] for l in ex["formatif"]["lignes"]]
    Xc = [l["texte"] for l in ex["certificatif"]["lignes"]]
    yc = [l["competence"] for l in ex["certificatif"]["lignes"]]
    ref_etiq = ref if all(isinstance(c.etiquette, str) for c in ref.competences) else None
    if ref_etiq is None:
        raise ValueError("dans l'atelier, les étiquettes du référentiel doivent être des textes (ex. C1)")
    res_f = examiner_classification(ref, yf, m.predire(Xf))
    decision = arret.evaluer(n_tours, res_f, lambda: examiner_classification(ref, yc, m.predire(Xc)))
    secondes = time.perf_counter() - t0
    j.ajouter("examen_formatif", "piolet", {"tour": n_tours, "etiquettes": len(lab), "score_min": round(res_f.score_min(), 4), "non_acquises": res_f.non_acquises(), "secondes": round(secondes, 3)})
    if decision.session is not None:
        j.ajouter("session_certification", "piolet", {"tour": n_tours, "session": decision.session, "reussie": decision.certifie, "score_min": round(decision.certificatif.score_min(), 4)})

    sortie: dict[str, Any] = {"tour": n_tours, "decision": decision.raison, "certifie": decision.certifie, "non_acquises": res_f.non_acquises()}
    if decision.arreter:
        j.ajouter("fin", "piolet", {"raison": decision.raison, "certifie": decision.certifie, "etiquettes": len(lab), "sessions": arret.sessions})
    else:
        deja_vus = {l["id"] for l in lab}
        candidats = [l for l in lire_csv(reservoir) if l.get("id") and l["id"] not in deja_vus and l.get("texte")]
        choix = proposer_lot(m, np.asarray([c["texte"] for c in candidats], dtype=object), lot, etiquettes_non_acquises=ref.etiquettes(res_f.non_acquises(arret.marge)))
        ecrire_csv(sortie_lot, [{"id": candidats[i]["id"], "texte": candidats[i]["texte"], "competence": "", "provenance": ""} for i in choix])
        sortie["a_etiqueter"] = str(sortie_lot)
        sortie["lot"] = len(choix)

    ancre = j.ancre()
    releve = {
        "type": "releve",
        "modele": {"nom": modele},
        "referentiel": ref.resume(),
        "certifie": decision.certifie,
        "raison": decision.raison,
        "sessions": arret.sessions,
        "formatif": res_f.en_dict(),
        "certificatif": decision.certificatif.en_dict() if decision.certificatif else None,
        "ticket": ticket_entrainement(len(lab), secondes_avant + secondes, sessions=arret.sessions).en_dict(),
        "provenance": {"controle": "garde de provenance Piolet", "acceptes": len(lab), "refuses": 0, "par_provenance": _compter(lab)},
        "journal": {"fichier": Path(journal).name, "entrees": ancre.entrees, "empreinte": ancre.empreinte, "signe": False},
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(sortie_releve).write_text(json.dumps(releve, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    sortie["releve"] = str(sortie_releve)
    return sortie


def _compter(lignes: list[dict[str, str]]) -> dict[str, int]:
    compte: dict[str, int] = {}
    for l in lignes:
        compte[l.get("provenance", "")] = compte.get(l.get("provenance", ""), 0) + 1
    return compte
