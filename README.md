# Prédiction du churn clients d'un opérateur télécom

Projet personnel de classification supervisée en Python (pandas, scikit-learn) : prédire quels clients risquent de quitter un opérateur télécom afin de cibler des actions de rétention.

## Données

- Dataset public **Telco Customer Churn** (IBM sample data), disponible sur Kaggle : <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- 7 043 clients, 21 colonnes (profil client, services souscrits, contrat, facturation).
- Cible : `Churn` (le client a quitté l'opérateur : Yes / No).
- Les données ne sont pas incluses dans ce dépôt : le fichier `WA_Fn-UseC_-Telco-Customer-Churn.csv` est à télécharger sur Kaggle.

## Méthode

1. **Nettoyage** : conversion de `TotalCharges` en numérique (11 valeurs vides, correspondant à des clients avec `tenure = 0`, supprimées), suppression de `customerID`, encodage de la cible en 0/1.
2. **Analyse exploratoire** : taux de churn par type de contrat et par ancienneté.
3. **Préparation** : encodage des variables catégorielles (`get_dummies`), séparation train 80 % / test 20 % stratifiée (`random_state=42`).
4. **Modèles** : régression logistique (avec `StandardScaler`), régression logistique avec `class_weight="balanced"`, random forest (200 arbres).
5. **Évaluation** sur le jeu de test : accuracy, matrice de confusion, précision, recall, AUC.
6. **Interprétation** : coefficients de la régression logistique.

## Résultats (jeu de test : 1 407 clients, dont 374 partis)

| Modèle | Accuracy | Recall (churn) | Précision (churn) | AUC |
|---|---|---|---|---|
| Régression logistique | 0,804 | 0,57 | 0,65 | ≈ 0,83 |
| Régression logistique, `class_weight="balanced"` | 0,726 | 0,80 | 0,49 | ≈ 0,83 |
| Random forest | 0,787 | 0,51 | 0,62 | ≈ 0,82 |

- La régression logistique obtient les meilleurs résultats sur ce jeu de données et reste simple à interpréter ; c'est le modèle retenu.
- `class_weight="balanced"` ne rend pas le modèle meilleur : il change le compromis. Il détecte davantage de clients qui partent (recall 0,80) au prix de davantage de fausses alertes (précision 0,49).
- Le choix entre les deux dépend du coût d'une offre de rétention par rapport à la perte d'un client.

### Facteurs associés au départ

- Risque plus élevé : contrat mensuel, faible ancienneté, internet en fibre optique.
- Risque plus faible : contrat d'un ou deux ans, forte ancienneté.
- Taux de churn observés : contrat mensuel ≈ 43 %, contrat d'un an ≈ 11 %, contrat de deux ans ≈ 3 %.

## Limites

- Dataset public d'exemple, pas des données réelles d'opérateur.
- Un seul découpage train/test, pas de validation croisée.
- Aucun réglage d'hyperparamètres.
- Les résultats décrivent des **associations**, pas des causes. L'efficacité d'une offre de rétention devrait être testée (test A/B).
- `tenure`, `MonthlyCharges` et `TotalCharges` sont corrélées entre elles : leurs coefficients individuels sont à interpréter avec prudence.

## Utilisation

```bash
pip install pandas scikit-learn matplotlib jupyter
```

1. Télécharger `WA_Fn-UseC_-Telco-Customer-Churn.csv` depuis Kaggle et le placer à côté du notebook.
2. Ouvrir `churn_prediction.ipynb` (Jupyter ou Google Colab) et exécuter toutes les cellules.

Dans Google Colab, le notebook propose de téléverser le fichier CSV automatiquement.

## Structure

```
.
├── churn_prediction.ipynb   # notebook complet (nettoyage, analyse, modèles, évaluation)
└── README.md
```

## Auteure

Manal Bouzeggaoui, étudiante en Master 2 Algorithmique et Modélisation à l'Interface des Sciences (AMIS), Université Paris-Saclay / UVSQ.
