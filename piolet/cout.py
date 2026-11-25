"""Ticket de coût : ce que l'entraînement a consommé, chaque ligne marquée « mesuré » ou « estimé ».

Un chiffre mesuré vient de la boucle elle-même (étiquettes achetées, époques, secondes
de calcul). Un chiffre estimé dépend d'une hypothèse, écrite sur la ligne (puissance
de la machine, temps d'étiquetage). On ne mélange jamais les deux sans le dire.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

MESURE = "mesuré"
ESTIME = "estimé"


@dataclass(frozen=True)
class Ligne:
    libelle: str
    valeur: float
    unite: str
    nature: str
    hypothese: str = ""


@dataclass
class Ticket:
    lignes: list[Ligne] = field(default_factory=list)

    def mesure(self, libelle: str, valeur: float, unite: str) -> "Ticket":
        self.lignes.append(Ligne(libelle, valeur, unite, MESURE))
        return self

    def estime(self, libelle: str, valeur: float, unite: str, hypothese: str) -> "Ticket":
        if not hypothese:
            raise ValueError("une estimation doit dire son hypothèse")
        self.lignes.append(Ligne(libelle, valeur, unite, ESTIME, hypothese))
        return self

    def en_dict(self) -> list[dict]:
        return [asdict(l) for l in self.lignes]

    def texte(self) -> str:
        sortie = []
        for l in self.lignes:
            valeur = f"{l.valeur:,.0f}".replace(",", " ") if float(l.valeur).is_integer() else f"{l.valeur:.3g}"
            suffixe = f"  ({l.hypothese})" if l.hypothese else ""
            sortie.append(f"{l.libelle:<34} {valeur:>12} {l.unite:<10} {l.nature}{suffixe}")
        return "\n".join(sortie)


def ticket_entrainement(
    etiquettes: int,
    secondes: float,
    *,
    epoques: int | None = None,
    sessions: int | None = None,
    puissance_w: float | None = None,
    secondes_par_etiquette: float | None = None,
) -> Ticket:
    """Ticket standard d'une trajectoire Piolet ; les estimations n'apparaissent que si on donne l'hypothèse."""
    t = Ticket()
    pluriel = lambda n, mot: mot if n == 1 else mot + "s"
    t.mesure("Étiquettes utilisées", etiquettes, pluriel(etiquettes, "étiquette"))
    if epoques is not None:
        t.mesure("Époques", epoques, pluriel(epoques, "époque"))
    if sessions is not None:
        t.mesure("Sessions de certification", sessions, pluriel(sessions, "session"))
    t.mesure("Calcul (entraînement et examens)", round(secondes, 3), "secondes")
    if secondes_par_etiquette is not None:
        t.estime("Temps d'étiquetage", round(etiquettes * secondes_par_etiquette / 3600, 2), "heures", f"{secondes_par_etiquette:g} s par étiquette")
    if puissance_w is not None:
        t.estime("Énergie de calcul", secondes * puissance_w / 3.6e6, "kWh", f"puissance moyenne {puissance_w:g} W")
    return t


# --------------------------------------------------------------------- LoRA
CIBLES = ("q", "k", "v", "o", "gate", "up", "down")


def matrices_standard(dim: int, cibles: Iterable[str], *, ffn: int | None = None, dim_kv: int | None = None) -> list[tuple[int, int]]:
    """Dimensions (entrée, sortie) des matrices adaptées dans une couche de transformeur.

    ``q`` et ``o`` : dim × dim ; ``k`` et ``v`` : dim × dim_kv (dim_kv < dim quand les têtes
    clé-valeur sont partagées) ; ``gate``, ``up`` : dim × ffn ; ``down`` : ffn × dim.
    """
    dim_kv = dim_kv or dim
    formes = {"q": (dim, dim), "k": (dim, dim_kv), "v": (dim, dim_kv), "o": (dim, dim)}
    if ffn:
        formes.update({"gate": (dim, ffn), "up": (dim, ffn), "down": (ffn, dim)})
    sortie = []
    for c in cibles:
        if c not in formes:
            raise ValueError(f"cible inconnue ou sans dimension : {c} (ffn requis pour gate, up, down)")
        sortie.append(formes[c])
    return sortie


def parametres_lora(couches: int, matrices: Sequence[tuple[int, int]], rang: int) -> int:
    """Paramètres entraînables ajoutés par LoRA (Hu et al., 2021).

    Pour une matrice d_entree × d_sortie, LoRA ajoute A (rang × d_entree) et B
    (d_sortie × rang), soit rang × (d_entree + d_sortie) paramètres, par couche.
    """
    if couches < 1 or rang < 1:
        raise ValueError("couches et rang doivent être positifs")
    return couches * sum(rang * (a + b) for a, b in matrices)


def octets_adaptateur(parametres: int, octets_par_parametre: int = 2) -> int:
    """Taille d'un adaptateur sur disque (2 octets par paramètre en 16 bits)."""
    return parametres * octets_par_parametre
