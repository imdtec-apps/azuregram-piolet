"""Banc d'essai public et reproductible : combien d'étiquettes et d'époques pour certifier ?

Données : les 1 797 images de chiffres manuscrits fournies avec scikit-learn.
Référentiel : 10 compétences (un chiffre chacune), seuil F1 ≥ 0,90 pour chacune.

Pour chaque graine, les images sont réparties par classe, sans recouvrement :

* examen formatif : 30 par classe (300), passé à chaque tour ;
* examen certificatif : 30 par classe (300), passé seulement en session ;
* test final : 30 par classe (300), jamais vu par la méthode, pour vérifier après coup ;
* réservoir : le reste (897), où l'on « achète » les étiquettes.

Expérience A (étiquettes) : même modèle, plusieurs façons de choisir les cas et de s'arrêter.
Expérience B (époques) : toutes les étiquettes, plusieurs façons d'arrêter l'entraînement.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import __version__
from .arret import ArretSurPreuve
from .cout import ticket_entrainement
from .examen import Resultat, examiner_classification
from .journal import Journal
from .modeles import MLP, NoyauLogistique
from .provenance import controler
from .referentiel import Referentiel
from .remediation import entrainer_par_epoques, remedier

REF_CHIFFRES = Path(__file__).parent / "donnees" / "chiffres.yaml"

LIBELLES_A = {
    "volume": "Volume : tout le réservoir étiqueté",
    "uniforme": "Hasard + arrêt Piolet",
    "incertitude_naif": "Incertitude + arrêt au premier succès",
    "piolet": "Piolet : incertitude + arrêt sur preuve",
    "piolet_marge0": "Piolet sans marge de sécurité",
    "ciblee": "Variante ciblée + arrêt sur preuve",
}
LIBELLES_B = {
    "fixe": "200 époques fixes",
    "arret_sklearn": "Arrêt précoce scikit-learn",
    "piolet": "Piolet : arrêt sur preuve",
    "piolet_marge0": "Piolet sans marge de sécurité",
}


@dataclass(frozen=True)
class Config:
    graines: int = 20
    n_formatif: int = 30
    n_certificatif: int = 30
    n_test: int = 30
    amorce_par_classe: int = 2
    lot: int = 10
    epoques_max: int = 200
    delai_etiquettes: int = 3
    delai_epoques: int = 10
    cachees: int = 128
    taux: float = 0.003


def charger_donnees() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits

    d = load_digits()
    return d.data / 16.0, d.target


def decouper(y: np.ndarray, graine: int, cfg: Config) -> tuple[np.ndarray, ...]:
    """Indices (formatif, certificatif, test, réservoir), stratifiés par classe."""
    rng = np.random.default_rng(graine)
    ordre = rng.permutation(len(y))
    F, C, T, P = [], [], [], []
    a, b, c = cfg.n_formatif, cfg.n_formatif + cfg.n_certificatif, cfg.n_formatif + cfg.n_certificatif + cfg.n_test
    for classe in np.unique(y):
        idx = ordre[y[ordre] == classe]
        F += list(idx[:a]); C += list(idx[a:b]); T += list(idx[b:c]); P += list(idx[c:])
    return tuple(np.array(v) for v in (F, C, T, P))


def _test(ref: Referentiel, predire: Callable, X: np.ndarray, y: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    res = examiner_classification(ref, y[T], predire(X[T]))
    return {
        "score_min": round(res.score_min(), 4),
        "score_moyen": round(res.score_moyen(), 4),
        "toutes_acquises": res.tout_acquis(),
        "sous_seuil": len(res.non_acquises()),
    }


def essai_etiquettes(ref: Referentiel, X: np.ndarray, y: np.ndarray, graine: int, cfg: Config, strategie: str) -> dict[str, Any]:
    F, C, T, P = decouper(y, graine, cfg)
    modele = NoyauLogistique(graine=graine).preparer(X[P])  # noyau ajusté sans étiquettes
    if strategie == "volume":
        t0 = time.perf_counter()
        modele.entrainer(X[P], y[P])
        cert = examiner_classification(ref, y[C], modele.predire(X[C]))
        return {
            "graine": graine, "etiquettes": len(P), "certifie": cert.tout_acquis(), "raison": "volume",
            "sessions": 1, "secondes": round(time.perf_counter() - t0, 4), "test": _test(ref, modele.predire, X, y, T),
        }
    rng = np.random.default_rng(10_000 + graine)
    yP = y[P]
    amorce = np.concatenate([rng.choice(np.flatnonzero(yP == k), cfg.amorce_par_classe, replace=False) for k in np.unique(yP)])
    if strategie == "incertitude_naif":
        arret, choix = ArretSurPreuve.naif(), "incertitude"
    elif strategie == "piolet_marge0":
        arret, choix = ArretSurPreuve.depuis(ref, delai=cfg.delai_etiquettes, marge=0.0), "incertitude"
    else:
        arret = ArretSurPreuve.depuis(ref, delai=cfg.delai_etiquettes)
        choix = {"piolet": "incertitude", "uniforme": "uniforme", "ciblee": "ciblee"}[strategie]
    traj = remedier(
        ref, modele, X[P], lambda idx: yP[idx], formatif=(X[F], y[F]), certificatif=(X[C], y[C]),
        amorce=amorce, lot=cfg.lot, strategie=choix, arret=arret, graine=graine,
    )
    return {
        "graine": graine, "etiquettes": traj.etiquettes, "certifie": traj.certifie, "raison": traj.raison,
        "sessions": traj.sessions, "secondes": round(traj.secondes, 4), "test": _test(ref, modele.predire, X, y, T),
    }


def essai_epoques(ref: Referentiel, X: np.ndarray, y: np.ndarray, graine: int, cfg: Config, regime: str) -> dict[str, Any]:
    F, C, T, P = decouper(y, graine, cfg)
    classes = np.unique(y)
    if regime == "fixe":
        m = MLP(classes, cachees=(cfg.cachees,), taux=cfg.taux, graine=graine)
        t0 = time.perf_counter()
        for _ in range(cfg.epoques_max):
            m.epoque(X[P], y[P])
        secondes, epoques, predire, sessions = time.perf_counter() - t0, cfg.epoques_max, m.predire, 1
        certifie, raison = examiner_classification(ref, y[C], m.predire(X[C])).tout_acquis(), "epoques_max"
    elif regime == "arret_sklearn":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(hidden_layer_sizes=(cfg.cachees,), learning_rate_init=cfg.taux, batch_size=64,
                            random_state=graine, early_stopping=True, max_iter=cfg.epoques_max)
        t0 = time.perf_counter()
        clf.fit(X[P], y[P])
        secondes, epoques, predire, sessions = time.perf_counter() - t0, int(clf.n_iter_), clf.predict, 1
        certifie, raison = examiner_classification(ref, y[C], clf.predict(X[C])).tout_acquis(), "arret_sklearn"
    elif regime in ("piolet", "piolet_marge0"):
        m = MLP(classes, cachees=(cfg.cachees,), taux=cfg.taux, graine=graine)
        arret = ArretSurPreuve.depuis(ref, delai=cfg.delai_epoques, **({"marge": 0.0} if regime == "piolet_marge0" else {}))
        traj = entrainer_par_epoques(ref, m, X[P], y[P], formatif=(X[F], y[F]), certificatif=(X[C], y[C]),
                                     epoques_max=cfg.epoques_max, arret=arret)
        secondes, epoques, predire, sessions = traj.secondes, traj.epoques, m.predire, traj.sessions
        certifie, raison = traj.certifie, traj.raison
    else:
        raise ValueError(regime)
    return {
        "graine": graine, "epoques": epoques, "certifie": bool(certifie), "raison": raison, "sessions": sessions,
        "secondes": round(secondes, 4), "test": _test(ref, predire, X, y, T),
    }


def _stat(valeurs: list[float]) -> dict[str, float]:
    v = np.asarray(valeurs, dtype=float)
    return {
        "mediane": float(np.median(v)), "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
        "moyenne": round(float(v.mean()), 4), "min": float(v.min()), "max": float(v.max()),
    }


def resumer(essais: list[dict[str, Any]], cle: str) -> dict[str, Any]:
    return {
        cle: _stat([e[cle] for e in essais]),
        "secondes": _stat([e["secondes"] for e in essais]),
        "taux_certification": round(float(np.mean([e["certifie"] for e in essais])), 4),
        "sessions_moyennes": round(float(np.mean([e["sessions"] for e in essais])), 2),
        "test_score_min_moyen": round(float(np.mean([e["test"]["score_min"] for e in essais])), 4),
        "test_score_moyen": round(float(np.mean([e["test"]["score_moyen"] for e in essais])), 4),
        "test_taux_toutes_acquises": round(float(np.mean([e["test"]["toutes_acquises"] for e in essais])), 4),
        "test_sous_seuil_moyen": round(float(np.mean([e["test"]["sous_seuil"] for e in essais])), 2),
    }


def releve_demonstration(ref: Referentiel, X: np.ndarray, y: np.ndarray, cfg: Config, dossier: Path, graine: int = 0) -> dict[str, Any]:
    """Une trajectoire Piolet complète, journalisée, pour le relevé de démonstration."""
    F, C, T, P = decouper(y, graine, cfg)
    journal_chemin = dossier / "journal-demo.jsonl"
    if journal_chemin.exists():
        journal_chemin.unlink()
    journal = Journal(journal_chemin)
    yP = y[P]
    prov = controler(({"id": f"digits-{int(i)}", "provenance": "public"} for i in P), ref.provenances_autorisees)
    journal.ajouter("referentiel", "piolet", {"id": ref.id, "version": ref.version, "empreinte": ref.empreinte()})
    journal.ajouter("provenance", "piolet", {"reservoir": len(P), **{k: v for k, v in prov.en_dict().items() if k != "refuses"}, "refuses": len(prov.refuses)})
    modele = NoyauLogistique(graine=graine).preparer(X[P])
    rng = np.random.default_rng(10_000 + graine)
    amorce = np.concatenate([rng.choice(np.flatnonzero(yP == k), cfg.amorce_par_classe, replace=False) for k in np.unique(yP)])
    traj = remedier(ref, modele, X[P], lambda idx: yP[idx], formatif=(X[F], y[F]), certificatif=(X[C], y[C]), amorce=amorce,
                    lot=cfg.lot, strategie="incertitude", arret=ArretSurPreuve.depuis(ref, delai=cfg.delai_etiquettes), graine=graine, journal=journal)
    ancre = journal.ancre()
    ancre.enregistrer(dossier / "ancre-demo.json")
    ticket = ticket_entrainement(traj.etiquettes, traj.secondes, sessions=traj.sessions)
    return {
        "type": "releve",
        "modele": {"nom": "Démonstration · noyau RBF (Nyström) + régression logistique", "graine": graine},
        "referentiel": ref.resume(),
        "certifie": traj.certifie,
        "raison": traj.raison,
        "sessions": traj.sessions,
        "reservoir": len(P),
        "formatif": traj.formatif.en_dict() if traj.formatif else None,
        "certificatif": traj.certificatif.en_dict() if traj.certificatif else None,
        "tours": [t.en_dict() for t in traj.tours],
        "ticket": ticket.en_dict(),
        "provenance": {"controle": "garde de provenance Piolet", "acceptes": len(prov.acceptes), "refuses": len(prov.refuses), "par_provenance": dict(prov.par_provenance)},
        "journal": {"fichier": journal_chemin.name, "entrees": ancre.entrees, "empreinte": ancre.empreinte, "signe": False},
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def lancer(graines: int = 20, dossier: str | Path = "banc", rapide: bool = False, afficher: Callable[[str], None] = print) -> dict[str, Any]:
    cfg = Config(graines=graines)
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    ref = Referentiel.charger(REF_CHIFFRES)
    X, y = charger_donnees()
    strategies = ["volume", "piolet", "incertitude_naif"] if rapide else list(LIBELLES_A)
    regimes = ["piolet"] if rapide else list(LIBELLES_B)
    debut = time.perf_counter()

    A: dict[str, Any] = {}
    for s in strategies:
        essais = [essai_etiquettes(ref, X, y, g, cfg, s) for g in range(graines)]
        A[s] = {"libelle": LIBELLES_A[s], "resume": resumer(essais, "etiquettes"), "essais": essais}
        r = A[s]["resume"]
        afficher(f"A {s:<17} étiquettes médiane {r['etiquettes']['mediane']:>5.0f}  certifié {r['taux_certification']:>4.0%}  test min {r['test_score_min_moyen']:.3f}  test OK {r['test_taux_toutes_acquises']:>4.0%}")
    B: dict[str, Any] = {}
    for g_ in regimes:
        essais = [essai_epoques(ref, X, y, g, cfg, g_) for g in range(graines)]
        B[g_] = {"libelle": LIBELLES_B[g_], "resume": resumer(essais, "epoques"), "essais": essais}
        r = B[g_]["resume"]
        afficher(f"B {g_:<17} époques médiane {r['epoques']['mediane']:>5.0f}  certifié {r['taux_certification']:>4.0%}  test min {r['test_score_min_moyen']:.3f}  test OK {r['test_taux_toutes_acquises']:>4.0%}")

    import sklearn

    resultats = {
        "type": "banc",
        "piolet": __version__,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duree_secondes": round(time.perf_counter() - debut, 1),
        "environnement": {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__, "machine": platform.machine()},
        "config": asdict(cfg),
        "referentiel": ref.resume(),
        "etiquettes": A,
        "epoques": B,
    }
    (dossier / "resultats.json").write_text(json.dumps(resultats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    releve = releve_demonstration(ref, X, y, cfg, dossier)
    (dossier / "releve-demo.json").write_text(json.dumps(releve, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return resultats
