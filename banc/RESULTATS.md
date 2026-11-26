# Banc d'essai Piolet · résultats

Généré par `piolet banc --graines 20` le 2026-11-25 (Piolet 0.1.0, Python 3.13.18, scikit-learn 1.9.1, NumPy 2.5.3). Durée : 137 s. Ce fichier est produit par le code : ne pas le modifier à la main.

## Protocole

- **Données** : les 1 797 images de chiffres manuscrits (8 × 8 pixels) fournies avec scikit-learn (`load_digits`).
- **Référentiel** : 10 compétences, une par chiffre, en deux blocs ; seuil F1 ≥ 0,90 pour **chaque** compétence (`piolet/donnees/chiffres.yaml`).
- **Répartition**, pour chacune des 20 graines, par classe et sans recouvrement : examen formatif 30 par classe, examen certificatif 30 par classe, test final 30 par classe (jamais vu par la méthode), réservoir pour le reste (897 images).
- **Expérience A (étiquettes)** : régression logistique sur noyau RBF approché (Nyström, ajusté sans étiquettes). Amorce de 2 étiquettes par classe, lots de 10, budget = tout le réservoir.
- **Expérience B (époques)** : perceptron à une couche cachée de 128 neurones (Adam, taux 0,003), toutes les étiquettes du réservoir, 200 époques au plus.
- **Arrêt sur preuve** : une session de certification s'ouvre quand l'examen formatif dépasse chaque seuil d'une marge de 0,03 ; 5 sessions au plus ; après un échec, attente de 3 tours (A) ou 10 époques (B), doublée à chaque nouvel échec ; une session finale a lieu si le budget s'épuise.
- **Mesures** : médiane et quartiles sur les graines ; part des modèles certifiés ; sur le test final, part des modèles dont les 10 compétences restent ≥ 0,90 et F1 le plus bas (moyenne).

## Expérience A · étiquettes utilisées

- **Volume : tout le réservoir étiqueté** : 897 étiquettes (médiane ; quartiles 897 à 897 ; moyenne 897). Certifiés : 95 % (sessions : 1,0 en moyenne). Test final tenu : 100 % ; F1 le plus bas : 0,947 en moyenne.
- **Hasard + arrêt Piolet** : 505 étiquettes (médiane ; quartiles 350 à 683 ; moyenne 523). Certifiés : 95 % (sessions : 1,8 en moyenne). Test final tenu : 80 % ; F1 le plus bas : 0,918 en moyenne.
- **Incertitude + arrêt au premier succès** : 100 étiquettes (médiane ; quartiles 90 à 110 ; moyenne 104). Arrêt décidé sur l'examen formatif seul, sans examen de certification. Test final tenu : 55 % ; F1 le plus bas : 0,891 en moyenne.
- **Piolet : incertitude + arrêt sur preuve** : 140 étiquettes (médiane ; quartiles 120 à 153 ; moyenne 141). Certifiés : 100 % (sessions : 1,25 en moyenne). Test final tenu : 85 % ; F1 le plus bas : 0,927 en moyenne.
- **Piolet sans marge de sécurité** : 120 étiquettes (médiane ; quartiles 110 à 140 ; moyenne 124). Certifiés : 100 % (sessions : 1,55 en moyenne). Test final tenu : 75 % ; F1 le plus bas : 0,913 en moyenne.
- **Variante ciblée + arrêt sur preuve** : 155 étiquettes (médiane ; quartiles 140 à 170 ; moyenne 189). Certifiés : 100 % (sessions : 1,35 en moyenne). Test final tenu : 70 % ; F1 le plus bas : 0,911 en moyenne.

## Expérience B · époques d'entraînement

- **200 époques fixes** : 200 époques (médiane ; quartiles 200 à 200 ; moyenne 200). Certifiés : 95 % (sessions : 1,0 en moyenne). Test final tenu : 85 % ; F1 le plus bas : 0,930 en moyenne.
- **Arrêt précoce scikit-learn** : 20 époques (médiane ; quartiles 18 à 22 ; moyenne 20). Certifiés : 20 % (sessions : 1,0 en moyenne). Test final tenu : 30 % ; F1 le plus bas : 0,869 en moyenne.
- **Piolet : arrêt sur preuve** : 46 époques (médiane ; quartiles 22 à 200 ; moyenne 97). Certifiés : 95 % (sessions : 1,65 en moyenne). Test final tenu : 85 % ; F1 le plus bas : 0,925 en moyenne.
- **Piolet sans marge de sécurité** : 24 époques (médiane ; quartiles 17 à 50 ; moyenne 52). Certifiés : 95 % (sessions : 2,0 en moyenne). Test final tenu : 80 % ; F1 le plus bas : 0,918 en moyenne.

## Lecture

- Piolet certifie avec 140 étiquettes au lieu de 897 (84 % de moins). Le modèle au volume va plus loin que le niveau requis (F1 le plus bas 0,947 contre 0,927) : Piolet s'arrête au niveau exigé par le référentiel.
- S'arrêter au premier examen réussi coûte moins (100 étiquettes) mais le niveau ne tient sur le test final que pour 55 % des modèles, contre 85 % avec l'examen de certification séparé.
- La marge de sécurité a un prix explicite : sans marge, 120 étiquettes et test tenu 75 % ; avec 0,03, 140 étiquettes et 85 %.
- Résultat négatif : la variante « ciblée », qui privilégie les cas des compétences en échec, ne fait pas mieux que l'incertitude seule (155 étiquettes, test tenu 70 %). Elle reste disponible, mais n'est pas la stratégie par défaut.
- Sur les époques, Piolet s'arrête après 46 époques (médiane) au lieu de 200 ; le niveau tient sur le test final pour 85 % des modèles, comme après 200 époques. Quand le modèle plafonne près du seuil, Piolet va jusqu'au budget et passe la session finale : il ne coûte alors pas plus que l'entraînement fixe. C'est pourquoi la moyenne (97 époques, 51 % de moins) est moins favorable que la médiane (77 % de moins).
- L'arrêt précoce de scikit-learn (validation interne) s'arrête tôt (20 époques) mais ne connaît pas le niveau exigé : 20 % de modèles certifiés, test tenu 30 %.

## Limites

- Un petit jeu de données public et des modèles simples. Le transfert vers un grand modèle de langage affiné par LoRA, sur un référentiel RNCP réel, est l'objet du programme de recherche.
- Une certification se fait à un seuil : un modèle certifié juste au-dessus peut passer sous le seuil sur un autre échantillon. La marge et le nombre d'items par compétence règlent ce risque.
- Les temps de calcul dépendent de la machine ; les comptes d'étiquettes et d'époques, non.

## Historique de la méthode

- Première version du banc, le même jour : délai constant entre deux sessions, sans session finale. Expérience A : 140 étiquettes (médiane), 100 % certifiés, test tenu 85 %. Expérience B : 40 époques (médiane) mais 55 % seulement de modèles certifiés, parce que les sessions s'épuisaient trop tôt ou que le budget finissait sans session. Correction retenue : délai doublé après chaque échec, et session finale en fin de budget. Les deux versions sont décrites pour que ce choix soit vérifiable.
