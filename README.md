# Projet_2_Groupe4_Veille-de-r-putation-multilingue
Analyse de sentiments et veille de réputation multilingue (FR/EN) - Projet NLP


#### 📊 Veille de Réputation Multilingue (Français / Anglais)

Système d'analyse des sentiments et de suivi de la e-réputation client (secteur bancaire) capable de classifier des avis en **français**, en **anglais** et en **code-switching (FR/EN)**, tout en générant un tableau de bord hebdomadaire pour les équipes communication et marketing. 

#### 🎯 Objectifs du Projet

1. **Classifier automatiquement** chaque avis client selon trois polarités : 0 (**Négatif**), 1 (**Neutre**), 2 (**Positif**).
2. **Comparer rigoureusement trois familles d'approches NLP** sur un jeu de test indépendant gelé : 

  * **Baselines sans réseau de neurones :** Lexique bilingue manuel & TF-IDF (n-grammes mots + caractères) avec Régression Logistique.
  * **Encodeur multilingue fine-tuné :** XLM-RoBERTa-base (fine-tuning léger des 3 dernières couches).
  * **LLM Zero-Shot :** Qwen2.5-1.5B-Instruct avec comparaison de 3 stratégies de prompting.
3. **Évaluer la robustesse** face aux cas complexes (sarcasme, ironie, mélange FR/EN) ainsi que le compromis **vitesse / coût** pour 1 000 avis.
4. **Restituer un tableau de bord** montrant l'évolution hebdomadaire des sentiments par langue et les mots-clés récurrents dans les avis négatifs.

#### 🗂️ Structure du Dépôt

Afficher le code

Masquer le code

Télécharger le diagramme au format SVG

ASCII

├── Notebook_01_Veille_Reputation.ipynb   # Notebook principal (Exploration, Modèles, Évaluation, Dashboard)
├── test_independant_gelé.csv             # Jeu de test gelé de 200 avis bancaires (100 FR + 100 EN)
├── annotation_membre1_fr.csv             # Fichier d'annotation indépendant (Annotateur 1 - FR)
├── annotation_membre2_fr.csv             # Fichier d'annotation indépendant (Annotateur 2 - FR)
├── annotation_membre3_en.csv             # Fichier d'annotation indépendant (Annotateur 3 - EN)
├── annotation_membre4_en.csv             # Fichier d'annotation indépendant (Annotateur 4 - EN)
├── Rapport_Projet_NLP.pdf                # Rapport détaillé (8 pages max)
└── README.md                             # Documentation du projet

Utilisez le code avec précaution.

Copier le texte du code dans le presse-papiers.

#### 📊 Données et Protocole

* **Données d'entraînement :** Corpus public *Tweet Sentiment Multilingual* (Cardiff NLP) — 1 839 tweets d'entraînement et 324 de validation par langue (classes parfaitement équilibrées).
* **Jeu de test gelé (test_independant_gelé.csv) :** 200 avis bancaires construits par l'équipe (100 FR, 100 EN), annotés en double aveugle avec un **Kappa de Cohen de 1,00**, et jamais utilisés durant l'entraînement ou le réglage des hyperparamètres.
* **Sous-ensemble difficile (18 avis) :** 10 messages sarcastiques/ironiques et 8 messages bilingues (*code-switching* FR/EN) repérés manuellement.
* **Prétraitement :** Suppression des URLs et mentions @user, conservation du texte des #hashtags, décodage des entités HTML, normalisation des apostrophes et réduction des répétitions excessives de lettres (*« enooooorme »* → *« enoorme »*).

#### ⚙️ Installation et Exécution

#### Sur Google Colab (Recommandé)

1. Ouvrir Notebook_01_Veille_Reputation.ipynb dans Google Colab.
2. Activer l'accélérateur matériel gratuit : **Exécution → Modifier le type d'exécution → T4 GPU**.
3. Exécuter toutes les cellules dans l'ordre (**Exécution → Tout exécuter**).

#### En local

bash

pip install pandas numpy matplotlib seaborn scikit-learn datasets transformers torch emoji scipy
jupyter notebook Notebook_01_Veille_Reputation.ipynb

Utilisez le code avec précaution.

Copier le texte du code dans le presse-papiers.

#### 📈 Résultats Clés (Jeu de Test Gelé — 200 avis)

ApprocheF1 Macro (FR)F1 Macro (EN)Erreur Sarcasme / Bilingue (18 avis)Temps / 1 000 avis
****Baseline 1 : Lexique manuel****
0,2970,224100,0 %~ 0,05 s
****Baseline 2 : TF-IDF + RegLog****
0,3680,30783,3 %~ 0,12 s
****Moderne 1 : XLM-R Fine-tuné****
*(Voir notebook)**(Voir notebook)**(Voir notebook)**(Voir notebook)*
****Moderne 2 : LLM Zero-Shot (v3)****
*(Voir notebook)**(Voir notebook)**(Voir notebook)**(Voir notebook)*

#### 💡 Démonstration Rapide

Le notebook inclut une fonction appelable en direct pour tester n'importe quel avis : 

python

analyser_avis_client("Le virement a encore failed, c'est really pénible ce service.")

Utilisez le code avec précaution.

Copier le texte du code dans le presse-papiers.

#### 🚀 Pistes pour la Mise en Production

* **Architecture hybride en cascade :** Filtrer les avis simples à haute confiance avec TF-IDF (coût nul, quasi instantané) et router uniquement les cas ambigus, bilingues ou ironiques vers XLM-RoBERTa.
* **Conformité RGPD :** Anonymisation automatique des données personnelles (noms, numéros de compte/téléphone) par NER avant stockage ou inférence.
* **Apprentissage continu :** Ré-entraînement périodique sur des avis bancaires réels corrigés par les équipes métier afin de réduire le décalage de domaine (*domain shift*).
