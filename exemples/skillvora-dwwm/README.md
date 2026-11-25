# Piolet SkillMatch · titre Développeur web et web mobile (RNCP37674)

Ce dossier prépare la certification de **Piolet SkillMatch**, le modèle de Skillvora qui relie une tâche de poste à une compétence du titre professionnel. Rien n'y est encore entraîné : les examens et les exemples sont **à rédiger par les formateurs**.

- `referentiel.yaml` : deux blocs, huit compétences, intitulés recopiés de la fiche RNCP37674 (France compétences). Le seuil de 0,85 et la taille des examens sont des propositions, à valider par l'équipe pédagogique.
- `examen_formatif.csv` et `examen_certificatif.csv` : 30 tâches par compétence et par examen, soit 480 au total. Colonnes : `id,texte,competence,provenance`.
- `etiquetes.csv` : les exemples d'entraînement déjà étiquetés, avec les mêmes colonnes.
- `reservoir.csv` : des tâches sans étiquette, extraites d'offres d'emploi réelles. Colonnes : `id,texte`.

## Règles de rédaction

1. **Une ligne, une tâche réelle**, écrite comme dans une offre d'emploi. Exemple de forme : « verbe + objet + contexte ». Pas de copie des intitulés du référentiel.
2. **Provenance obligatoire** : `formateur` ou `expert-metier`. Une tâche écrite ou reformulée par un modèle d'IA tiers est refusée par la garde de provenance, et le tour s'arrête.
3. **Deux formateurs étiquettent chaque tâche de l'examen sans se consulter.** Ils comparent ensuite leurs réponses et règlent les désaccords en discutant. On note l'accord avant discussion : le kappa de Cohen.
4. **L'examen certificatif reste scellé.** Il n'est jamais montré au modèle, ni utilisé pour choisir des exemples.

## Vérifier, puis lancer un tour

```bash
piolet referentiel valider exemples/skillvora-dwwm/referentiel.yaml   # compte les tâches rédigées
piolet tour --referentiel exemples/skillvora-dwwm/referentiel.yaml \
            --etiquetes exemples/skillvora-dwwm/etiquetes.csv \
            --reservoir exemples/skillvora-dwwm/reservoir.csv \
            --journal journal-dwwm.jsonl
```

Tant que les examens ne sont pas complets, `piolet tour` refuse de démarrer. Ensuite, chaque tour écrit `a_etiqueter.csv`, les tâches où le modèle hésite le plus, jusqu'à la certification.
