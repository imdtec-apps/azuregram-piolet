"""Boucles d'entraînement, greffon Hugging Face (logique), journal, atelier, relevé, CLI."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from piolet import ArretSurPreuve, Journal, Referentiel, entrainer_par_epoques, proposer_lot, remedier
from piolet.banc import REF_CHIFFRES, charger_donnees, decouper, Config
from piolet.cli import main
from piolet.examen import Note, Resultat
from piolet.hf import PioletCallback
from piolet.journal import Ancre
from piolet.modeles import NoyauLogistique, charger_modele
from piolet.releve import rendre_html, resultats_markdown


# ---------------------------------------------------------------- remédiation
class ModeleConnu:
    """Modèle factice : probabilités fixées à l'avance, pour tester le choix des cas."""

    classes = np.array([0, 1])

    def __init__(self, proba):
        self._p = np.asarray(proba, dtype=float)

    def proba(self, X):
        return self._p[np.asarray(X, dtype=int)]


def test_proposer_lot_prend_les_cas_les_plus_hesitants():
    m = ModeleConnu([[0.99, 0.01], [0.5, 0.5], [0.7, 0.3], [0.55, 0.45]])
    assert list(proposer_lot(m, np.arange(4), 2)) == [1, 3]


def test_remedier_certifie_avant_le_budget():
    ref = Referentiel.charger(REF_CHIFFRES)
    X, y = charger_donnees()
    F, C, T, P = decouper(y, 0, Config())
    m = NoyauLogistique(graine=0).preparer(X[P])
    rng = np.random.default_rng(1)
    amorce = np.concatenate([rng.choice(np.flatnonzero(y[P] == k), 2, replace=False) for k in range(10)])
    traj = remedier(ref, m, X[P], lambda i: y[P][i], formatif=(X[F], y[F]), certificatif=(X[C], y[C]), amorce=amorce, lot=10)
    assert traj.certifie and traj.etiquettes < len(P) / 2
    assert traj.tours[-1].decision.session is not None


def test_remedier_budget_epuise_passe_la_session_finale():
    ref = Referentiel.charger(REF_CHIFFRES)
    X, y = charger_donnees()
    F, C, T, P = decouper(y, 0, Config())
    m = NoyauLogistique(graine=0).preparer(X[P])
    amorce = np.concatenate([np.flatnonzero(y[P] == k)[:2] for k in range(10)])
    traj = remedier(ref, m, X[P], lambda i: y[P][i], formatif=(X[F], y[F]), certificatif=(X[C], y[C]), amorce=amorce, lot=10, budget=30)
    assert traj.etiquettes == 30 and traj.raison in ("budget_epuise", "certifie_fin_de_budget") and traj.sessions == 1


class ModeleQuiProgresse:
    """Prédit juste une part croissante des cas, époque après époque."""

    def __init__(self):
        self.epoques = 0

    def epoque(self, X, y):
        self.epoques += 1
        self._y = np.asarray(y)

    def predire(self, X):
        X = np.asarray(X)
        bon = min(1.0, 0.5 + 0.1 * self.epoques)
        return np.where(X[:, 1] < bon, X[:, 0], 1 - X[:, 0])


def test_entrainer_par_epoques_s_arrete_des_la_preuve():
    ref = Referentiel.depuis_dict({"id": "e", "titre": "E", "tache": "classification", "seuil": 0.9,
                                   "certification": {"marge": 0, "delai": 1},
                                   "blocs": [{"id": "B", "competences": [{"id": "A", "etiquette": 0}, {"id": "Z", "etiquette": 1}]}]})
    rng = np.random.default_rng(0)
    Xe = np.column_stack([rng.integers(0, 2, 400), rng.random(400)])
    m = ModeleQuiProgresse()
    traj = entrainer_par_epoques(ref, m, Xe, Xe[:, 0], formatif=(Xe[:200], Xe[:200, 0]), certificatif=(Xe[200:], Xe[200:, 0]), epoques_max=50)
    assert traj.certifie and 4 <= traj.epoques <= 6  # 90 % de bons cas vers la 4e époque


# ------------------------------------------------------- greffon Hugging Face
def test_callback_arrete_le_trainer_quand_le_modele_est_certifie():
    scores = iter([0.5, 0.7, 0.95, 0.99])
    examen = lambda model: Resultat((Note("A", "B", next(scores), 0.9, 10),))
    cb = PioletCallback(ArretSurPreuve(marge=0.0, delai=1), examen, lambda model: Resultat((Note("A", "B", 0.93, 0.9, 10),)))
    control = SimpleNamespace(should_training_stop=False)
    for epoque in (1.0, 2.0, 3.0):
        cb.on_epoch_end(None, SimpleNamespace(epoch=epoque), control, model=object())
        if control.should_training_stop:
            break
    assert control.should_training_stop and cb.decision.certifie and cb.historique[-1]["epoque"] == 3


# ---------------------------------------------------------------------- journal
def test_journal_chaine_ancre_et_signature(tmp_path):
    j = Journal(tmp_path / "j.jsonl", horloge=lambda: "2026-01-01T00:00:00+00:00")
    for k in range(4):
        j.ajouter("evenement", "test", {"k": k})
    assert j.verifier() == (True, None)
    ancre = j.ancre().signer(b"secret")
    assert ancre.signature_valide(b"secret") and not ancre.signature_valide(b"autre")
    lignes = (tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines()
    alteree = json.loads(lignes[1]); alteree["donnees"]["k"] = 99
    (tmp_path / "j.jsonl").write_text("\n".join([lignes[0], json.dumps(alteree)] + lignes[2:]) + "\n", encoding="utf-8")
    assert j.verifier() == (False, 1)
    (tmp_path / "j.jsonl").write_text("\n".join(lignes[:2]) + "\n", encoding="utf-8")  # fin du journal supprimée
    assert j.verifier() == (True, None) and j.verifier(ancre) == (False, 2)


# ----------------------------------------------------------------------- atelier
def _csv(chemin, lignes, colonnes=("id", "texte", "competence", "provenance")):
    with open(chemin, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(colonnes)); w.writeheader(); w.writerows(lignes)


def test_atelier_un_tour_puis_certification(tmp_path):
    (tmp_path / "ref.yaml").write_text(
        "id: a\ntitre: A\ntache: classification\nseuil: 0.8\nprovenances_autorisees: [formateur]\n"
        "certification: {marge: 0, delai: 1}\nexamens: {formatif: f.csv, certificatif: c.csv, items_par_competence: 5}\n"
        "blocs:\n  - id: B\n    competences:\n      - {id: C1, etiquette: C1}\n      - {id: C2, etiquette: C2}\n", encoding="utf-8")
    mots = {"C1": "base de données table requête sql", "C2": "maquette écran interface couleur"}
    def lignes(prefixe, n):
        return [{"id": f"{prefixe}{c}{i}", "texte": f"{mots[c]} {prefixe}{i}", "competence": c, "provenance": "formateur"} for c in mots for i in range(n)]
    _csv(tmp_path / "f.csv", lignes("f", 5)); _csv(tmp_path / "c.csv", lignes("c", 5))
    _csv(tmp_path / "lab.csv", lignes("l", 3))
    _csv(tmp_path / "res.csv", [{"id": f"r{i}", "texte": f"{mots['C1' if i % 2 else 'C2']} r{i}"} for i in range(30)], colonnes=("id", "texte"))
    from piolet.atelier import tour
    r = tour(tmp_path / "ref.yaml", tmp_path / "lab.csv", tmp_path / "res.csv", tmp_path / "j.jsonl",
             lot=4, sortie_lot=tmp_path / "lot.csv", sortie_releve=tmp_path / "releve.json")
    assert r["certifie"] and json.loads((tmp_path / "releve.json").read_text(encoding="utf-8"))["type"] == "releve"
    assert tour(tmp_path / "ref.yaml", tmp_path / "lab.csv", tmp_path / "res.csv", tmp_path / "j.jsonl")["message"].startswith("modèle déjà certifié")
    # une provenance tierce dans les exemples bloque le tour
    _csv(tmp_path / "lab2.csv", lignes("m", 3)[:1] + [{"id": "x", "texte": "t", "competence": "C1", "provenance": "sortie-modele-tiers"}])
    from piolet.provenance import ProvenanceRefusee
    with pytest.raises(ProvenanceRefusee):
        tour(tmp_path / "ref.yaml", tmp_path / "lab2.csv", tmp_path / "res.csv", tmp_path / "j2.jsonl")


# -------------------------------------------------------------- relevé et CLI
def test_page_embarque_les_donnees_sans_casser_le_script():
    page = rendre_html({"type": "releve", "modele": {"nom": "</script><b>x"}}, None)
    assert "<\\/script><b>x" in page and "/*__PIOLET_DONNEES__*/" not in page


def test_cli(tmp_path, capsys):
    assert main(["cout-lora", "--couches", "32", "--dim", "4096", "--cibles", "q,v", "--rang", "8"]) == 0
    assert "4 194 304" in capsys.readouterr().out
    assert main(["referentiel", "valider", str(REF_CHIFFRES)]) == 0
    dwwm = Path(__file__).resolve().parents[1] / "exemples" / "skillvora-dwwm" / "referentiel.yaml"
    assert main(["referentiel", "valider", str(dwwm)]) == 0
    assert "0/240" in capsys.readouterr().out
    assert "texte-logistique" in [m for m in __import__("piolet.modeles", fromlist=["x"]).modeles_disponibles()]
    assert charger_modele("noyau-logistique", gamma=0.2).gamma == 0.2


def test_banc_rapide_et_compte_rendu(tmp_path):
    assert main(["banc", "--graines", "1", "--rapide", "--sortie", str(tmp_path)]) == 0
    res = json.loads((tmp_path / "resultats.json").read_text(encoding="utf-8"))
    assert res["etiquettes"]["piolet"]["resume"]["etiquettes"]["mediane"] < 897
    assert (tmp_path / "RESULTATS.md").read_text(encoding="utf-8").startswith("# Banc d'essai Piolet")
    assert "piolet-donnees" in (tmp_path / "releve.html").read_text(encoding="utf-8")
    assert main(["journal", "verifier", str(tmp_path / "journal-demo.jsonl"), "--ancre", str(tmp_path / "ancre-demo.json")]) == 0
    assert resultats_markdown(res)
