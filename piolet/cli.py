"""Ligne de commande ``piolet``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

from . import __version__


def _cmd_referentiel(args: argparse.Namespace) -> int:
    from .referentiel import Referentiel, ReferentielInvalide

    try:
        ref = Referentiel.charger(args.fichier)
    except ReferentielInvalide as e:
        print(e, file=sys.stderr)
        return 1
    comps = ref.competences
    print(f"Référentiel valide : {ref.titre} ({ref.id}, version {ref.version})")
    if ref.source:
        print(f"  source : {ref.source}")
    print(f"  tâche : {ref.tache} · {len(ref.blocs)} bloc(s) · {len(comps)} compétence(s)")
    seuils = sorted({c.seuil for c in comps})
    print(f"  seuils : {', '.join(f'{s:g}' for s in seuils)}")
    c = ref.certification
    print(f"  certification : marge {c.marge:g}, {c.sessions_max} sessions au plus, délai {c.delai} tour(s) × {c.croissance:g} après chaque échec")
    print(f"  provenances autorisées : {', '.join(sorted(ref.provenances_autorisees))}")
    print(f"  empreinte : {ref.empreinte()}")
    manquants = ref.a_rediger()
    if manquants:
        print(f"  à rédiger : {len(manquants)} item(s) d'examen")
    if ref.brut.get("examens"):
        from .atelier import examens

        try:
            ex = examens(ref, Path(args.fichier).parent)
        except (OSError, Exception) as e:  # fichiers absents ou illisibles
            print(f"  examens : illisibles ({e})")
            return 0
        for nom in ("formatif", "certificatif"):
            total = sum(ex[nom]["par_competence"].values())
            attendu = ex["minimum"] * len(comps)
            etat = "complet" if ex[nom]["complet"] else "incomplet"
            print(f"  examen {nom} : {total}/{attendu} items rédigés ({etat}, {ex['minimum']} par compétence au minimum)")
    return 0


def _cmd_banc(args: argparse.Namespace) -> int:
    from .banc import lancer
    from .releve import rendre_html, resultats_markdown

    warnings.filterwarnings("ignore")
    dossier = Path(args.sortie)
    resultats = lancer(graines=args.graines, dossier=dossier, rapide=args.rapide)
    (dossier / "RESULTATS.md").write_text(resultats_markdown(resultats), encoding="utf-8")
    releve = json.loads((dossier / "releve-demo.json").read_text(encoding="utf-8"))
    page = Path(args.page) if args.page else dossier / "releve.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(rendre_html(releve, resultats), encoding="utf-8")
    print(f"Résultats : {dossier / 'resultats.json'} · {dossier / 'RESULTATS.md'} · page : {page}")
    return 0


def _cmd_releve(args: argparse.Namespace) -> int:
    from .releve import rendre_html

    releve = json.loads(Path(args.releve).read_text(encoding="utf-8")) if args.releve else None
    banc = json.loads(Path(args.banc).read_text(encoding="utf-8")) if args.banc else None
    Path(args.sortie).write_text(rendre_html(releve, banc), encoding="utf-8")
    print(f"Page écrite : {args.sortie}")
    return 0


def _cmd_cout_lora(args: argparse.Namespace) -> int:
    from .cout import matrices_standard, octets_adaptateur, parametres_lora

    cibles = [c.strip() for c in args.cibles.split(",") if c.strip()]
    matrices = matrices_standard(args.dim, cibles, ffn=args.ffn, dim_kv=args.dim_kv)
    p = parametres_lora(args.couches, matrices, args.rang)
    taille = octets_adaptateur(p, args.octets)
    print(f"Paramètres entraînables : {p:,}".replace(",", " "))
    if args.total:
        print(f"Part du modèle : {100 * p / args.total:.3g} %")
    print(f"Taille de l'adaptateur : {taille / 1e6:.1f} Mo ({args.octets} octets par paramètre)")
    return 0


def _cmd_journal(args: argparse.Namespace) -> int:
    from .journal import Ancre, Journal

    j = Journal(args.fichier)
    ancre = Ancre.charger(args.ancre) if args.ancre else None
    ok, index = j.verifier(ancre)
    if not ok:
        print(f"Journal ALTÉRÉ : première anomalie à l'entrée {index}")
        return 2
    message = f"Journal intact : {len(j.lire())} entrées"
    if ancre is not None:
        message += ", conforme à l'ancre"
        cle = os.environ.get(args.cle_env, "")
        if ancre.signature:
            if not cle:
                message += " (signature non vérifiée : variable " + args.cle_env + " absente)"
            elif ancre.signature_valide(cle.encode("utf-8")):
                message += ", signature valide"
            else:
                print("Signature de l'ancre INVALIDE")
                return 2
        else:
            message += " (ancre non signée)"
    print(message)
    return 0


def _cmd_provenance(args: argparse.Namespace) -> int:
    from .provenance import controler
    from .referentiel import Referentiel

    ref = Referentiel.charger(args.referentiel)
    chemin = Path(args.fichier)
    if chemin.suffix.lower() == ".csv":
        from .atelier import lire_csv

        exemples = lire_csv(chemin)
    else:
        exemples = [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]
    bilan = controler(exemples, ref.provenances_autorisees)
    print(f"{len(bilan.acceptes)} accepté(s), {len(bilan.refuses)} refusé(s) · {dict(bilan.par_provenance)}")
    for i, p, r in bilan.refuses[:20]:
        print(f"  refusé : {i} ({p or 'sans provenance'}) : {r}")
    return 0 if bilan.conforme else 3


def _cmd_tour(args: argparse.Namespace) -> int:
    from .atelier import tour

    warnings.filterwarnings("ignore")
    r = tour(args.referentiel, args.etiquetes, args.reservoir, args.journal, lot=args.lot, modele=args.modele, sortie_lot=args.lot_sortie, sortie_releve=args.releve)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    return 0


def _cmd_modeles(_: argparse.Namespace) -> int:
    from .modeles import modeles_disponibles

    print("\n".join(modeles_disponibles()))
    return 0


def construire() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="piolet", description="Piolet : entraîner sur preuve.")
    p.add_argument("--version", action="version", version=f"piolet {__version__}")
    sp = p.add_subparsers(dest="commande", required=True)

    r = sp.add_parser("referentiel", help="valider un référentiel")
    rs = r.add_subparsers(dest="action", required=True)
    rv = rs.add_parser("valider", help="vérifier le schéma, les seuils, les provenances et les examens")
    rv.add_argument("fichier")
    rv.set_defaults(f=_cmd_referentiel)

    b = sp.add_parser("banc", help="lancer le banc d'essai public (chiffres manuscrits)")
    b.add_argument("--graines", type=int, default=20)
    b.add_argument("--sortie", default="banc")
    b.add_argument("--page", default=None, help="chemin de la page HTML (par défaut : <sortie>/releve.html)")
    b.add_argument("--rapide", action="store_true", help="moins de variantes, pour un test")
    b.set_defaults(f=_cmd_banc)

    rl = sp.add_parser("releve", help="produire la page HTML d'un relevé et/ou d'un banc")
    rl.add_argument("releve", nargs="?", default=None)
    rl.add_argument("--banc", default=None)
    rl.add_argument("-o", "--sortie", required=True)
    rl.set_defaults(f=_cmd_releve)

    c = sp.add_parser("cout-lora", help="paramètres entraînables et taille d'un adaptateur LoRA")
    c.add_argument("--couches", type=int, required=True)
    c.add_argument("--dim", type=int, required=True)
    c.add_argument("--dim-kv", type=int, default=None)
    c.add_argument("--ffn", type=int, default=None)
    c.add_argument("--cibles", default="q,v")
    c.add_argument("--rang", type=int, default=8)
    c.add_argument("--total", type=float, default=None, help="paramètres du modèle complet")
    c.add_argument("--octets", type=int, default=2)
    c.set_defaults(f=_cmd_cout_lora)

    j = sp.add_parser("journal", help="vérifier un journal chaîné")
    js = j.add_subparsers(dest="action", required=True)
    jv = js.add_parser("verifier")
    jv.add_argument("fichier")
    jv.add_argument("--ancre", default=None)
    jv.add_argument("--cle-env", default="PIOLET_CLE", help="variable d'environnement qui contient la clé HMAC")
    jv.set_defaults(f=_cmd_journal)

    pv = sp.add_parser("provenance", help="contrôler la provenance d'exemples (CSV ou JSONL)")
    pv.add_argument("fichier")
    pv.add_argument("--referentiel", required=True)
    pv.set_defaults(f=_cmd_provenance)

    t = sp.add_parser("tour", help="un tour de la boucle avec des formateurs")
    t.add_argument("--referentiel", required=True)
    t.add_argument("--etiquetes", required=True)
    t.add_argument("--reservoir", required=True)
    t.add_argument("--journal", required=True)
    t.add_argument("--lot", type=int, default=20)
    t.add_argument("--modele", default="texte-logistique")
    t.add_argument("--lot-sortie", default="a_etiqueter.csv")
    t.add_argument("--releve", default="releve.json")
    t.set_defaults(f=_cmd_tour)

    m = sp.add_parser("modeles", help="lister les modèles disponibles (intégrés et greffons)")
    m.set_defaults(f=_cmd_modeles)
    return p


def main(argv: list[str] | None = None) -> int:
    args = construire().parse_args(argv)
    return int(args.f(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
