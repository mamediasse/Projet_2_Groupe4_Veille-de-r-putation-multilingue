# Projet_2_Groupe4_Veille-de-r-putation-multilingue
Analyse de sentiments et veille de réputation multilingue (FR/EN) - Projet NLP


#### 📊 Veille de Réputation Multilingue (Français / Anglais) — Projet 2 (Groupe 4)

[](https://colab.research.google.com/github/mamediasse/veille-reputation-multilingue-nlp/blob/main/Projet_2_Groupe_4.ipynb) 

Système d'analyse des sentiments et de veille de réputation client (secteur bancaire) capable de classifier des avis en **français**, en **anglais** et en **code-switching (FR/EN)**, accompagné d'un tableau de bord hebdomadaire pour les équipes communication et marketing. 

#### 🎯 Objectifs du Projet

1. **Classifier automatiquement** chaque avis client selon trois polarités : 0 (**Négatif**), 1 (**Neutre**), 2 (**Positif**).
2. **Comparer rigoureusement trois familles d'approches NLP** sur un jeu de test indépendant gelé : 

  * **Baselines sans réseau de neurones :** Lexique bilingue manuel & TF-IDF (n-grammes mots + caractères) avec Régression Logistique.
  * **Encodeur multilingue fine-tuné :** XLM-RoBERTa-base (fine-tuning léger des 3 dernières couches).
  * **LLM Zero-Shot :** Qwen2.5-1.5B-Instruct avec comparaison de 3 versions de prompts.
3. **Évaluer la robustesse** face aux cas complexes (sarcasme, ironie, mélange FR/EN) ainsi que le compromis **vitesse / coût** pour 1 000 avis.
4. **Restituer un tableau de bord** montrant l'évolution hebdomadaire des sentiments par langue et les mots les plus fréquents dans les avis négatifs.

#### 🗂️ Contenu du Dépôt

FichierDescription
****Projet_2_Groupe_4.ipynb****
Notebook principal complet (Exploration, Prétraitement, Modèles, Évaluation, Analyse d'erreurs, Tableau de bord et Démonstration)
****README.md****
Documentation principale du dépôt
****Raaport_Projet NLP.pdf****
Rapport détaillé du projet (contexte, méthodologie, résultats, analyse d'erreurs et bilan critique)
****Video_Demonstration_1.mp4****
Démonstration vidéo du fonctionnement du système (Partie 1)
****Video_Demonstration_2.mp4****
Démonstration vidéo du tableau de bord et des prédictions en direct (Partie 2)
****annotation_membre1_fr.csv****
Annotations indépendantes — Membre 1 (Français)
****annotation_membre2_fr.csv****
Annotations indépendantes — Membre 2 (Français)
****annotation_membre3_en.csv****
Annotations indépendantes — Membre 3 (Anglais)
****annotation_membre4_en.csv****
Annotations indépendantes — Membre 4 (Anglais)
****test_independant_gelé.csv****
Jeu de test gelé de 200 avis bancaires construits par l'équipe (100 FR + 100 EN)

#### 📊 Données et Protocole

* **Données d'entraînement et de validation :** Corpus public *Tweet Sentiment Multilingual* (Cardiff NLP) — 1 839 tweets d'entraînement et 324 de validation par langue (classes parfaitement équilibrées).
* **Jeu de test gelé (test_independant_gelé.csv) :** 200 avis bancaires construits par l'équipe (100 FR, 100 EN), annotés en double aveugle avec un **Kappa de Cohen de 1,00**, et jamais utilisés durant l'entraînement ou le réglage des hyperparamètres.
* **Sous-ensemble difficile (18 avis) :** 10 messages sarcastiques/ironiques et 8 messages bilingues (*code-switching* FR/EN) repérés à la main.
* **Prétraitement :** Suppression des URLs et mentions @user, conservation du texte des #hashtags, décodage des entités HTML, normalisation des apostrophes et réduction des répétitions excessives de lettres (*« enooooorme »* → *« enoorme »*).

#### ⚙️ Installation et Exécution

#### Sur Google Colab (Recommandé)

1. Cliquez sur le badge **Open in Colab** en haut de cette page (ou importez Projet_2_Groupe_4.ipynb).
2. Activez l'accélérateur matériel gratuit : **Exécution → Modifier le type d'exécution → T4 GPU**.
3. Lancez toutes les cellules : **Exécution → Tout exécuter**.

#### En local

bash

pip install pandas numpy matplotlib seaborn scikit-learn datasets transformers torch emoji scipy
jupyter notebook Projet_2_Groupe_4.ipynb

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

Le notebook intègre une fonction appelable en direct (analyser_avis_client) ainsi que deux vidéos de démonstration (Video_Demonstration_1.mp4 et Video_Demonstration_2.mp4) : 

python

analyser_avis_client("Le virement a encore failed, c'est really pénible ce service.")

Utilisez le code avec précaution.

Copier le texte du code dans le presse-papiers.

#### 🚀 Pistes pour la Mise en Production

* **Architecture hybride en cascade :** Filtrer les avis simples à haute confiance avec TF-IDF (coût nul, quasi instantané) et router uniquement les cas ambigus, bilingues ou sarcastiques vers XLM-RoBERTa.
* **Conformité RGPD :** Anonymisation automatique des données personnelles (noms, numéros de compte/téléphone) par NER avant stockage ou inférence.
* **Apprentissage continu :** Ré-entraînement périodique sur des avis bancaires réels corrigés par les équipes métier afin de réduire le décalage de domaine (*domain shift*).
