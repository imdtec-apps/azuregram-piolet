"""Référentiel, examen, arrêt sur preuve, provenance, promotion, coût."""

import math

import numpy as np
import pytest

from piolet import (
    ArretSurPreuve,
    ExamenIncomplet,
    ProvenanceRefusee,
    Referentiel,
    ReferentielInvalide,
    controler,
    decider_promotion,
    examiner_classification,
    examiner_generation,
    exiger,
    matrices_standard,
    parametres_lora,
    verifier_reponse,
)
from piolet.banc import REF_CHIFFRES
from piolet.cout import Ticket
from piolet.examen import Note, Resultat


def ref_simple(**cert):
    d = {
        "id": "t", "titre": "Test", "tache": "classification", "seuil": 0.8,
        "certification": {"marge": 0.0, "sessions_max": 3, "delai": 2, "croissance": 2, **cert},
        "provenances_autorisees": ["formateur"],
        "blocs": [{"id": "B", "titre": "Bloc", "competences": [{"id": "A", "etiquette": "a"}, {"id": "Z", "etiquette": "z"}]}],
    }
    return Referentiel.depuis_dict(d)


def resultat(*scores, seuil=0.8):
    return Resultat(tuple(Note(f"K{i}", "B", s, seuil, 10) for i, s in enumerate(scores)))


# ------------------------------------------------------------------ référentiel
def test_referentiel_chiffres():
    ref = Referentiel.charger(REF_CHIFFRES)
    assert len(ref.competences) == 10 and [b.id for b in ref.blocs] == ["B1", "B2"]
    assert ref.etiquettes(["C3", "C9"]) == [3, 9]
    assert ref.empreinte() == Referentiel.charger(REF_CHIFFRES).empreinte()
    assert ref.certification.croissance == 2


def test_referentiel_invalide_liste_tous_les_problemes():
    d = {
        "id": "x", "titre": "X", "tache": "classification", "seuil": 1.5,
        "blocs": [{"id": "B", "competences": [{"id": "C", "etiquette": 1}, {"id": "C"}]}],
    }
    with pytest.raises(ReferentielInvalide) as e:
        Referentiel.depuis_dict(d)
    texte = " ".join(e.value.problemes)
    assert "seuil" in texte and "double" in texte and "etiquette" in texte


def test_generation_items_a_rediger_et_provenance():
    d = {
        "id": "g", "titre": "G", "tache": "generation", "provenances_autorisees": ["formateur"],
        "blocs": [{"id": "B", "competences": [{"id": "C", "items": [
            {"id": "i1", "consigne": "[à rédiger par un formateur]", "attendu": "x", "provenance": "formateur"}]}]}],
    }
    ref = Referentiel.depuis_dict(d)
    assert ref.a_rediger() == ["i1"]
    with pytest.raises(ExamenIncomplet):
        examiner_generation(ref, lambda consigne: "x")
    d["blocs"][0]["competences"][0]["items"][0]["provenance"] = "api-tierce"
    with pytest.raises(ReferentielInvalide):
        Referentiel.depuis_dict(d)


# ----------------------------------------------------------------------- examen
def test_examen_f1_par_competence():
    ref = ref_simple()
    vrai = ["a", "a", "a", "z", "z"]
    pred = ["a", "a", "z", "z", "z"]
    res = examiner_classification(ref, vrai, pred)
    assert res.note("A").score == pytest.approx(2 * 2 / (2 * 2 + 0 + 1))  # F1 = 0,8
    assert res.note("Z").score == pytest.approx(2 * 2 / (2 * 2 + 1 + 0))
    assert res.tout_acquis() and not res.tout_acquis(marge=0.05)


def test_verifier_reponse():
    assert verifier_reponse("Réponse : Requête PARAMÉTRÉE", "requête paramétrée")
    assert verifier_reponse("42", "42", "egal") and not verifier_reponse("420", "42", "egal")
    assert verifier_reponse("SELECT * FROM t WHERE id = ?", r"where\s+id\s*=\s*\?", "regex")


def test_examen_generation():
    d = {"id": "g", "titre": "G", "tache": "generation", "seuil": 0.5, "provenances_autorisees": ["formateur"],
         "blocs": [{"id": "B", "competences": [{"id": "C", "items": [
             {"id": "i1", "consigne": "2+2", "attendu": "4", "verification": "egal", "provenance": "formateur"},
             {"id": "i2", "consigne": "3+3", "attendu": "6", "verification": "egal", "provenance": "formateur"}]}]}]}
    ref = Referentiel.depuis_dict(d)
    res = examiner_generation(ref, lambda q: {"2+2": "4", "3+3": "7"}[q])
    assert res.note("C").score == 0.5 and res.tout_acquis()


# ------------------------------------------------------------------------ arrêt
def test_arret_sur_preuve_sessions_delai_croissant():
    a = ArretSurPreuve(marge=0.05, sessions_max=3, delai=2, croissance=2)
    reussi, echoue = resultat(0.9, 0.9), resultat(0.9, 0.7)
    assert a.evaluer(0, resultat(0.82, 0.9), lambda: reussi).raison == "formatif_insuffisant"  # marge non atteinte
    d = a.evaluer(1, resultat(0.9, 0.9), lambda: echoue)
    assert (d.raison, d.session, d.arreter) == ("session_echouee", 1, False)
    assert a.evaluer(2, resultat(0.9, 0.9), lambda: reussi).raison == "delai"  # attente 2 tours
    d = a.evaluer(3, resultat(0.9, 0.9), lambda: echoue)
    assert d.session == 2 and a.attente == 4  # l'attente double
    assert a.evaluer(6, resultat(0.9, 0.9), lambda: reussi).raison == "delai"
    d = a.evaluer(7, resultat(0.9, 0.9), lambda: echoue)
    assert (d.raison, d.arreter, d.certifie) == ("sessions_epuisees", True, False)


def test_arret_certifie_et_naif_et_session_finale():
    a = ArretSurPreuve(marge=0.0, sessions_max=2, delai=1)
    d = a.evaluer(0, resultat(0.85, 0.85), lambda: resultat(0.81, 0.9))
    assert d.certifie and d.arreter and d.raison == "certifie"
    n = ArretSurPreuve.naif()
    assert n.evaluer(0, resultat(0.8, 0.8), lambda: pytest.fail("pas de certification en mode naïf")).raison == "formatif_reussi"
    f = ArretSurPreuve(sessions_max=1)
    assert f.session_finale(lambda: resultat(0.9, 0.9)).raison == "certifie_fin_de_budget"
    assert f.session_finale(lambda: resultat(0.9, 0.9)).raison == "budget_epuise"  # plus de session


# ------------------------------------------------------------------- provenance
def test_garde_de_provenance():
    ex = [{"id": 1, "provenance": "formateur"}, {"id": 2, "provenance": "api-tierce"}, {"id": 3}, {"id": 4, "provenance": "inconnu"}]
    b = controler(ex, ["formateur", "api-tierce"])  # une provenance interdite ne s'autorise pas
    assert b.acceptes == ["1"] and [r[0] for r in b.refuses] == ["2", "3", "4"]
    with pytest.raises(ProvenanceRefusee):
        exiger(ex, ["formateur"])
    assert exiger(ex[:1], ["formateur"]).conforme


# -------------------------------------------------------------------- promotion
def test_garde_de_promotion():
    en_place = resultat(0.85, 0.90)
    assert decider_promotion(resultat(0.86, 0.91), en_place).promouvoir
    bloque = decider_promotion(resultat(0.95, 0.88), en_place)  # moyenne meilleure, recul sur K1
    assert not bloque.promouvoir and "K1" in bloque.raisons[0]
    assert decider_promotion(resultat(0.95, 0.88), en_place, tolerance=0.03).promouvoir
    assert not decider_promotion(resultat(0.70, 0.95)).promouvoir


# ------------------------------------------------------------------------- coût
def test_parametres_lora_cas_connu():
    # modèle de 7 milliards (32 couches, dimension 4 096), rang 8 sur q et v : 4 194 304 paramètres
    assert parametres_lora(32, matrices_standard(4096, ["q", "v"]), 8) == 4_194_304
    # têtes clé-valeur partagées : k et v plus petites
    assert matrices_standard(2048, ["k"], dim_kv=256) == [(2048, 256)]
    with pytest.raises(ValueError):
        matrices_standard(4096, ["up"])  # ffn requis


def test_ticket_exige_une_hypothese_pour_estimer():
    t = Ticket().mesure("Étiquettes", 140, "étiquettes")
    with pytest.raises(ValueError):
        t.estime("Énergie", 0.1, "kWh", "")
    t.estime("Énergie", 0.1, "kWh", "65 W")
    assert [l["nature"] for l in t.en_dict()] == ["mesuré", "estimé"]
    assert math.isclose(t.lignes[1].valeur, 0.1)


def test_page_lisible_sans_javascript():
    """Les chiffres du banc et du relevé figurent dans le HTML lui-même, pour les lecteurs sans JavaScript."""
    import json
    from pathlib import Path

    from piolet.releve import rendre_html

    racine = Path(__file__).resolve().parent.parent / "banc"
    banc = json.loads((racine / "resultats.json").read_text(encoding="utf-8"))
    releve = json.loads((racine / "releve-demo.json").read_text(encoding="utf-8"))
    page = rendre_html(releve, banc)
    assert '<div id="banc-contenu"><div class="statique">' in page
    assert '<div class="carte" id="carte-releve"><div class="statique">' in page
    assert "140 étiquettes" in page and "897 étiquettes" in page
    assert "Moyenne" in page
