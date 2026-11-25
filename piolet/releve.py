"""Relevé : la page HTML autonome qui présente un relevé de compétences et un banc d'essai.

La page (``gabarit/releve.html``) ne charge rien depuis le réseau : les données sont
incluses dans la page, et un relevé local peut y être ouvert sans être envoyé.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

GABARIT = Path(__file__).parent / "gabarit" / "releve.html"
_MARQUE = "/*__PIOLET_DONNEES__*/"


def alleger_banc(banc: dict[str, Any]) -> dict[str, Any]:
    """Garde les résumés, retire le détail des essais (il reste dans resultats.json)."""
    leger = {k: v for k, v in banc.items() if k not in ("etiquettes", "epoques")}
    for groupe in ("etiquettes", "epoques"):
        leger[groupe] = {k: {"libelle": v["libelle"], "resume": v["resume"]} for k, v in banc.get(groupe, {}).items()}
    return leger


def rendre_html(releve: dict[str, Any] | None = None, banc: dict[str, Any] | None = None) -> str:
    gabarit = GABARIT.read_text(encoding="utf-8")
    donnees: dict[str, Any] = {}
    if releve:
        donnees["releve"] = releve
    if banc:
        donnees["banc"] = alleger_banc(banc)
    charge = json.dumps(donnees, ensure_ascii=False).replace("</", "<\\/")
    return gabarit.replace(_MARQUE, charge)


def _pc(x: float) -> str:
    return f"{round(100 * x):d} %"


def _nb(x: float) -> str:
    """Entier arrondi au demi supérieur (comme la page HTML), avec espace des milliers."""
    return f"{int(math.floor(x + 0.5)):,}".replace(",", " ")


def _f3(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def resultats_markdown(banc: dict[str, Any]) -> str:
    """Compte rendu lisible de ``resultats.json`` (généré, ne pas éditer à la main)."""
    cfg, env = banc["config"], banc["environnement"]
    L = [
        "# Banc d'essai Piolet · résultats",
        "",
        f"Généré par `piolet banc --graines {cfg['graines']}` le {banc['date'][:10]} (Piolet {banc['piolet']}, Python {env['python']}, "
        f"scikit-learn {env['scikit-learn']}, NumPy {env['numpy']}). Durée : {banc['duree_secondes']:.0f} s. Ce fichier est produit par le code : ne pas le modifier à la main.",
        "",
        "## Protocole",
        "",
        "- **Données** : les 1 797 images de chiffres manuscrits (8 × 8 pixels) fournies avec scikit-learn (`load_digits`).",
        "- **Référentiel** : 10 compétences, une par chiffre, en deux blocs ; seuil F1 ≥ 0,90 pour **chaque** compétence (`piolet/donnees/chiffres.yaml`).",
        f"- **Répartition**, pour chacune des {cfg['graines']} graines, par classe et sans recouvrement : examen formatif {cfg['n_formatif']} par classe, "
        f"examen certificatif {cfg['n_certificatif']} par classe, test final {cfg['n_test']} par classe (jamais vu par la méthode), réservoir pour le reste (897 images).",
        f"- **Expérience A (étiquettes)** : régression logistique sur noyau RBF approché (Nyström, ajusté sans étiquettes). Amorce de {cfg['amorce_par_classe']} étiquettes par classe, lots de {cfg['lot']}, budget = tout le réservoir.",
        f"- **Expérience B (époques)** : perceptron à une couche cachée de {cfg['cachees']} neurones (Adam, taux {str(cfg['taux']).replace('.', ',')}), toutes les étiquettes du réservoir, {cfg['epoques_max']} époques au plus.",
        f"- **Arrêt sur preuve** : une session de certification s'ouvre quand l'examen formatif dépasse chaque seuil d'une marge de 0,03 ; 5 sessions au plus ; après un échec, "
        f"attente de {cfg['delai_etiquettes']} tours (A) ou {cfg['delai_epoques']} époques (B), doublée à chaque nouvel échec ; une session finale a lieu si le budget s'épuise.",
        "- **Mesures** : médiane et quartiles sur les graines ; part des modèles certifiés ; sur le test final, part des modèles dont les 10 compétences restent ≥ 0,90 et F1 le plus bas (moyenne).",
        "",
    ]
    for titre, groupe, cle, unite in (("Expérience A · étiquettes utilisées", "etiquettes", "etiquettes", "étiquettes"), ("Expérience B · époques d'entraînement", "epoques", "epoques", "époques")):
        L += [f"## {titre}", ""]
        for k, v in banc[groupe].items():
            r = v["resume"]
            s = r[cle]
            certif = ("Arrêt décidé sur l'examen formatif seul, sans examen de certification. " if r["sessions_moyennes"] == 0 else
                      f"Certifiés : {_pc(r['taux_certification'])} (sessions : {str(r['sessions_moyennes']).replace('.', ',')} en moyenne). ")
            L.append(
                f"- **{v['libelle']}** : {_nb(s['mediane'])} {unite} (médiane ; quartiles {_nb(s['q1'])} à {_nb(s['q3'])}). "
                + certif +
                f"Test final tenu : {_pc(r['test_taux_toutes_acquises'])} ; F1 le plus bas : {_f3(r['test_score_min_moyen'])} en moyenne."
            )
        L.append("")
    A, B = banc["etiquettes"], banc.get("epoques", {})
    L += ["## Lecture", ""]
    if "piolet" in A and "volume" in A:
        p, v = A["piolet"]["resume"], A["volume"]["resume"]
        L.append(f"- Piolet certifie avec {_nb(p['etiquettes']['mediane'])} étiquettes au lieu de {_nb(v['etiquettes']['mediane'])} "
                 f"({_pc(1 - p['etiquettes']['mediane'] / v['etiquettes']['mediane'])} de moins). Le modèle au volume va plus loin que le niveau requis "
                 f"(F1 le plus bas {_f3(v['test_score_min_moyen'])} contre {_f3(p['test_score_min_moyen'])}) : Piolet s'arrête au niveau exigé par le référentiel.")
    if "incertitude_naif" in A and "piolet" in A:
        n, p = A["incertitude_naif"]["resume"], A["piolet"]["resume"]
        L.append(f"- S'arrêter au premier examen réussi coûte moins ({_nb(n['etiquettes']['mediane'])} étiquettes) mais le niveau ne tient sur le test final que pour "
                 f"{_pc(n['test_taux_toutes_acquises'])} des modèles, contre {_pc(p['test_taux_toutes_acquises'])} avec l'examen de certification séparé.")
    if "piolet_marge0" in A and "piolet" in A:
        m, p = A["piolet_marge0"]["resume"], A["piolet"]["resume"]
        L.append(f"- La marge de sécurité a un prix explicite : sans marge, {_nb(m['etiquettes']['mediane'])} étiquettes et test tenu {_pc(m['test_taux_toutes_acquises'])} ; "
                 f"avec 0,03, {_nb(p['etiquettes']['mediane'])} étiquettes et {_pc(p['test_taux_toutes_acquises'])}.")
    if "ciblee" in A and "piolet" in A:
        c = A["ciblee"]["resume"]
        L.append(f"- Résultat négatif : la variante « ciblée », qui privilégie les cas des compétences en échec, ne fait pas mieux que l'incertitude seule "
                 f"({_nb(c['etiquettes']['mediane'])} étiquettes, test tenu {_pc(c['test_taux_toutes_acquises'])}). Elle reste disponible, mais n'est pas la stratégie par défaut.")
    if "piolet" in B and "fixe" in B:
        p, f = B["piolet"]["resume"], B["fixe"]["resume"]
        comparaison = (f"comme après {_nb(f['epoques']['mediane'])} époques" if p['test_taux_toutes_acquises'] == f['test_taux_toutes_acquises']
                       else f"contre {_pc(f['test_taux_toutes_acquises'])} après {_nb(f['epoques']['mediane'])} époques")
        L.append(f"- Sur les époques, Piolet s'arrête après {_nb(p['epoques']['mediane'])} époques (médiane) au lieu de {_nb(f['epoques']['mediane'])} ; "
                 f"le niveau tient sur le test final pour {_pc(p['test_taux_toutes_acquises'])} des modèles, {comparaison}. Quand le modèle plafonne près du seuil, "
                 "Piolet va jusqu'au budget et passe la session finale : il ne coûte alors pas plus que l'entraînement fixe.")
    if "arret_sklearn" in B:
        a = B["arret_sklearn"]["resume"]
        L.append(f"- L'arrêt précoce de scikit-learn (validation interne) s'arrête tôt ({_nb(a['epoques']['mediane'])} époques) mais ne connaît pas le niveau exigé : "
                 f"{_pc(a['taux_certification'])} de modèles certifiés, test tenu {_pc(a['test_taux_toutes_acquises'])}.")
    L += [
        "",
        "## Limites",
        "",
        "- Un petit jeu de données public et des modèles simples. Le transfert vers un grand modèle de langage affiné par LoRA, sur un référentiel RNCP réel, est l'objet du programme de recherche.",
        "- Une certification se fait à un seuil : un modèle certifié juste au-dessus peut passer sous le seuil sur un autre échantillon. La marge et le nombre d'items par compétence règlent ce risque.",
        "- Les temps de calcul dépendent de la machine ; les comptes d'étiquettes et d'époques, non.",
        "",
        "## Historique de la méthode",
        "",
        "- Première version du banc, le même jour : délai constant entre deux sessions, sans session finale. Expérience A : 140 étiquettes (médiane), "
        "100 % certifiés, test tenu 85 %. Expérience B : 40 époques (médiane) mais 55 % seulement de modèles certifiés, parce que les sessions s'épuisaient trop tôt "
        "ou que le budget finissait sans session. Correction retenue : délai doublé après chaque échec, et session finale en fin de budget. Les deux versions sont décrites pour que ce choix soit vérifiable.",
        "",
    ]
    return "\n".join(L)
