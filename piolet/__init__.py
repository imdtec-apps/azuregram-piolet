"""Piolet — entraîner sur preuve.

Un greffon d'entraînement qui pilote un modèle comme on forme un professionnel :
un référentiel de compétences, un examen par compétence, une remédiation sur ce qui
manque, et l'arrêt dès que la preuve est faite, consignés dans un relevé signé.
"""

__version__ = "0.1.0"

from .arret import ArretSurPreuve, Decision
from .cout import Ticket, matrices_standard, octets_adaptateur, parametres_lora, ticket_entrainement
from .examen import ExamenIncomplet, Note, Resultat, examiner_classification, examiner_generation, verifier_reponse
from .journal import Ancre, Journal
from .promotion import DecisionPromotion, decider as decider_promotion
from .provenance import INTERDITES, Bilan, ProvenanceRefusee, controler, exiger
from .referentiel import Bloc, Certification, Competence, Item, Referentiel, ReferentielInvalide
from .remediation import STRATEGIES, Trajectoire, entrainer_par_epoques, proposer_lot, remedier

__all__ = [
    "__version__",
    "ArretSurPreuve", "Decision",
    "Ticket", "matrices_standard", "octets_adaptateur", "parametres_lora", "ticket_entrainement",
    "ExamenIncomplet", "Note", "Resultat", "examiner_classification", "examiner_generation", "verifier_reponse",
    "Ancre", "Journal",
    "DecisionPromotion", "decider_promotion",
    "INTERDITES", "Bilan", "ProvenanceRefusee", "controler", "exiger",
    "Bloc", "Certification", "Competence", "Item", "Referentiel", "ReferentielInvalide",
    "STRATEGIES", "Trajectoire", "entrainer_par_epoques", "proposer_lot", "remedier",
]
