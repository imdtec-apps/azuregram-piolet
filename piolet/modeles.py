"""Modèles branchables.

Piolet ne fournit pas de modèle : il pilote celui qu'on lui donne. Un modèle est un
objet qui sait ``entrainer(X, y)``, ``proba(X)`` et ``predire(X)`` ; un modèle entraîné
par époques sait aussi ``epoque(X, y)``. Trois adaptateurs scikit-learn sont fournis.

D'autres modèles se branchent par le groupe de points d'entrée ``piolet.modeles``
(voir ``charger_modele``) : c'est le mécanisme de greffon.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Modele(Protocol):
    classes: np.ndarray

    def entrainer(self, X: Any, y: Any) -> None: ...

    def proba(self, X: Any) -> np.ndarray: ...

    def predire(self, X: Any) -> np.ndarray: ...


@runtime_checkable
class ModeleEpoques(Modele, Protocol):
    def epoque(self, X: Any, y: Any) -> None: ...


class NoyauLogistique:
    """Régression logistique sur un noyau RBF approché (méthode de Nyström).

    Le noyau est ajusté sur des données **sans étiquettes** (``preparer``) ; seule la
    régression logistique utilise les étiquettes.
    """

    def __init__(self, gamma: float = 0.1, composantes: int = 400, C: float = 30.0, graine: int = 0):
        self.gamma, self.composantes, self.C, self.graine = gamma, composantes, C, graine
        self._noyau = None
        self._clf = None
        self.classes = np.array([])

    def preparer(self, X_sans_etiquettes: Any) -> "NoyauLogistique":
        from sklearn.kernel_approximation import Nystroem

        n = min(self.composantes, len(X_sans_etiquettes))
        self._noyau = Nystroem(gamma=self.gamma, n_components=n, random_state=self.graine).fit(X_sans_etiquettes)
        return self

    def _z(self, X: Any) -> np.ndarray:
        if self._noyau is None:
            raise RuntimeError("appeler preparer() avec des données sans étiquettes avant d'entraîner")
        return self._noyau.transform(X)

    def entrainer(self, X: Any, y: Any) -> None:
        from sklearn.linear_model import LogisticRegression

        self._clf = LogisticRegression(max_iter=5000, C=self.C).fit(self._z(X), y)
        self.classes = self._clf.classes_

    def proba(self, X: Any) -> np.ndarray:
        return self._clf.predict_proba(self._z(X))

    def predire(self, X: Any) -> np.ndarray:
        return self._clf.predict(self._z(X))


class MLP:
    """Perceptron multicouche entraîné époque par époque (``partial_fit`` de scikit-learn)."""

    def __init__(self, classes: Any, cachees: tuple[int, ...] = (128,), taux: float = 0.003, lot: int = 64, graine: int = 0):
        from sklearn.neural_network import MLPClassifier

        self.classes = np.asarray(classes)
        self._clf = MLPClassifier(hidden_layer_sizes=cachees, learning_rate_init=taux, batch_size=lot, random_state=graine)
        self.epoques = 0

    def epoque(self, X: Any, y: Any) -> None:
        self._clf.partial_fit(X, y, classes=self.classes)
        self.epoques += 1

    def entrainer(self, X: Any, y: Any, epoques: int = 200) -> None:
        for _ in range(epoques):
            self.epoque(X, y)

    def proba(self, X: Any) -> np.ndarray:
        return self._clf.predict_proba(X)

    def predire(self, X: Any) -> np.ndarray:
        return self._clf.predict(X)


class TexteLogistique:
    """Classifieur de textes courts : TF-IDF (mots et caractères) puis régression logistique.

    Pensé pour relier une tâche de poste à une compétence d'un référentiel (Piolet SkillMatch).
    """

    def __init__(self, C: float = 10.0):
        self.C = C
        self._pipe = None
        self.classes = np.array([])

    def entrainer(self, X: Any, y: Any) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline, make_union

        vecteurs = make_union(
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
        )
        self._pipe = make_pipeline(vecteurs, LogisticRegression(max_iter=5000, C=self.C)).fit(list(X), list(y))
        self.classes = self._pipe.classes_

    def proba(self, X: Any) -> np.ndarray:
        return self._pipe.predict_proba(list(X))

    def predire(self, X: Any) -> np.ndarray:
        return self._pipe.predict(list(X))


INTEGRES = {"noyau-logistique": NoyauLogistique, "mlp": MLP, "texte-logistique": TexteLogistique}


def modeles_disponibles() -> list[str]:
    noms = set(INTEGRES)
    noms.update(ep.name for ep in entry_points(group="piolet.modeles"))
    return sorted(noms)


def charger_modele(nom: str, **parametres: Any) -> Any:
    """Instancie un modèle intégré, ou un modèle déclaré par un greffon tiers (point d'entrée)."""
    for ep in entry_points(group="piolet.modeles"):
        if ep.name == nom:
            return ep.load()(**parametres)
    if nom in INTEGRES:
        return INTEGRES[nom](**parametres)
    raise KeyError(f"modèle inconnu : {nom} (disponibles : {', '.join(modeles_disponibles())})")
