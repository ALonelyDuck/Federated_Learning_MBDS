
<div class="title">Apprentissage Fédéré</div>
<div class="separator"></div>
<div class="subtitle">Implémentation et analyse sur le jeu de données MNIST</div>

<div class="authors">
<p><strong>Par</strong></p>
<p>Clément COLIN</p>
<p>Enzo ROCAMORA</p>
<p>Thomas CHOUBRAC</p>
</div>

<p>Avril 2025<p>

<div class="page"/>

## Sommaire

## [1. Introduction](#1-introduction)

## [2. Jeu de données MNIST](#2-jeu-de-données-mnist)

## [3. Architecture et méthodologie](#3-architecture-et-méthodologie)

## [4. Distribution des données](#4-distribution-des-données)

## [5. Algorithmes d'agrégation](#5-algorithmes-dagrégation)

## [6. Expérimentations et paramètres](#6-expérimentations-et-paramètres)

## [7. Résultats et analyse](#7-résultats-et-analyse)

## [8. Sécurité et confidentialité dans l'apprentissage fédéré](#8-sécurité-et-confidentialité-dans-lapprentissage-fédéré)

## [9. Conclusion et perspectives](#9-conclusion-et-perspectives)

## [10. Références bibliographiques](#10-références-bibliographiques)

<div class="page"/>

## Introduction

L'apprentissage fédéré répond à plusieurs enjeux. D'une part, il respecte la confidentialité des données, sujet sensible depuis l'entrée en vigueur du RGPD. D'autre part, il permet de tirer parti de données distribuées géographiquement sans avoir à les centraliser, ce qui peut s'avérer impossible pour des raisons techniques, légales ou d'évolution constante des données.

Notre objectif a été d'implémenter et d'analyser différentes variantes d'apprentissage fédéré sur le jeu de données MNIST, afin de comprendre les impacts de divers paramètres sur les performances des modèles.

## Fondements théoriques de l'apprentissage fédéré

### Principe général

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>

L'apprentissage fédéré se déroule généralement en trois étapes principales :

1. Un serveur central initialise un modèle global et le distribue aux clients
2. Chaque client entraîne le modèle sur ses données locales
3. Les mises à jour des modèles locaux sont agrégées par le serveur central

Mathématiquement, l'objectif est de résoudre un problème d'optimisation de la forme :

$$\min_{w \in \mathbb{R}^d} f(w)$$

où $f(w)$ est décomposable sous forme d'une somme finie :

$$f(w) = \frac{1}{N} \sum_{i=1}^{N} f_i(w)$$

Dans le contexte du machine learning, la fonction $f_i$ représente la fonction de perte pour le point de données $i$ :

$$f_i(w) = \text{loss}(x_i, y_i; w)$$

Lorsque les données sont réparties entre $K$ clients, nous pouvons réécrire l'objectif comme :

$$f(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w)$$

où $F_k(w) = \frac{1}{n_k} \sum_{i \in P_k} f_i(w)$ est la fonction objectif locale du client $k$, $P_k$ est l'ensemble des indices des points de données du client $k$, $n_k = |P_k|$ est le nombre de données du client $k$, et $n = \sum_{k=1}^{K} n_k$ est le nombre total de données.

<div class="page"/>

## Jeu de données MNIST

Pour notre étude, nous avons utilisé le jeu de données MNIST, qui contient des images en niveaux de gris de chiffres manuscrits (de 0 à 9). Voici ses caractéristiques principales :

- 37 800 images d'entraînement
- 4 200 images de test
- Dimensions des images : 28×28 pixels (en niveaux de gris)

La distribution des classes dans l'ensemble d'entraînement est relativement équilibrée, avec environ 10% des images pour chaque chiffre. Cette répartition équilibrée nous a permis d'expérimenter facilement différentes stratégies de distribution entre clients.

Pour charger et prétraiter ces données, nous avons implémenté le module `fl_dataquest.py` qui convertit les images en tenseurs normalisés et crée des pipelines TensorFlow efficaces :

```python
def get_data(img_path = '../Mnist/trainingSet/trainingSet/', verbose = 0):
    # Chargement et prétraitement des images MNIST
    image_paths = list(list_images(img_path))
    il, ll = load_and_preprocess(image_paths, verbose=10000)
    
    # Binarisation des labels (one-hot encoding)
    lb = skl.preprocessing.LabelBinarizer()
    ll = lb.fit_transform(ll)
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(il, ll, 
                                                       test_size=0.1, 
                                                       random_state=19)
    
    return X_train, X_test, y_train, y_test, X_train[0].shape
```

## Tentative d'application sur d'autres jeux de données

Nous avons également exploré l'utilisation d'autres jeux de données, notamment le dataset Auto-MPG (Session 2). Mais nous avons rencontré plusieurs obstacles. La taille réduite du jeu de données rendait difficile une distribution significative entre plusieurs clients. De plus, les caractéristiques numériques continues présentaient des distributions très hétérogènes, ce qui entraînait des problèmes de convergence dans le cadre de l'apprentissage fédéré. Nos tests préliminaires ont montré des écarts de performance considérables entre les différents clients, selon la distribution des données.

<div class="page"/>

## Architecture du modèle et méthodologie expérimentale

### Prérequis et installation

Pour exécuter notre implémentation d'apprentissage fédéré, vous aurez besoin d'installer les bibliothèques Python suivantes.

```bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python pandas seaborn psutil
```

### Structure du projet

Notre projet est organisé en plusieurs modules Python :

- `fl_model.py` : Définition du modèle neuronal
- `fl_types.py` : Implémentation de l'apprentissage fédéré horizontal
- `fl_dataquest.py` : Chargement et prétraitement des données MNIST
- `data_partition.py` : Fonctions pour la répartition des données entre clients
- `aggregation.py` : Algorithmes d'agrégation (FedAvg, FedSGD, FedProx)
- `utils.py` : Fonctions utilitaires
- `run_all.py` : Script principal pour exécuter toutes les expériences

### Exécution des expériences

Pour lancer l'ensemble des expériences, exécutez simplement le script principal :

```bash
python run_all.py
```

Ce script va :
1. Charger les données MNIST
2. Créer un modèle centralisé de référence
3. Exécuter toutes les configurations d'apprentissage fédéré définies
4. Évaluer les performances de chaque configuration
5. Générer des visualisations et sauvegarder les résultats

L'exécution complète peut prendre plusieurs heures selon votre matériel, car de nombreuses configurations sont testées (nombre de clients, de rounds, d'époques locales, différents algorithmes, etc.).

<div class="page"/>

### Fichiers générés

1. **Résultats CSV** :
   - `federated_learning_results.csv` : Résumé des performances finales de chaque configuration
   - `federated_learning_detailed_results.csv` : Résultats détaillés round par round

2. **Visualisations** :
   - Dossier `plots/` : Graphiques comparatifs des différentes configurations

3. **Visualisations de progression** :
   - Dossier `progression_plots/` : Visualisations PCA de la trajectoire des paramètres

4. **Historique des paramètres** :
   - Dossier `parameter_history/` : Fichiers pickle contenant l'historique des paramètres pour chaque configuration

### Modèle neuronal

Pour notre étude, nous avons utilisé un réseau de neurones multicouche (MLP) implémenté dans la classe `MyModel` du module `fl_model.py` :

```python
class MyModel():
    def __init__(self, input_shape, nbclasses):
        model = Sequential()
        model.add(Input(input_shape))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dense(nbclasses))
        model.add(Activation("softmax"))

        self.model = model
        self.input_shape = input_shape
        self.classes = nbclasses

        self.loss_fn = 'categorical_crossentropy'
        self.model.compile(optimizer="SGD", loss=self.loss_fn, metrics=["accuracy"])
```

Cette architecture relativement simple est suffisante pour obtenir de bons résultats sur MNIST tout en restant légère en termes de calcul, ce qui est important dans un contexte d'apprentissage fédéré où les ressources des clients peuvent être limitées.

### Mécanisme d'apprentissage fédéré

Le cœur de notre implémentation repose sur la fonction `horizontal_federated_learning` du module `fl_types.py`, qui coordonne l'entraînement entre le serveur central et les clients :

```python
def horizontal_federated_learning(edges, central_model, input_shape, num_classes, 
                                 edge_epochs, test_data, aggregation_fn, verbose=0):
    '''Apprentissage fédéré horizontal (HFL)'''
    central_weights = central_model.get_weights()
    scaled_local_weight_list = []
    
    # Pour chaque client
    edges_names = list(edges.keys())
    random.shuffle(edges_names)
    
    for client_name in edges_names:
        # Obtenir les données du client
        client_data = edges[client_name]
        
        # Créer et configurer le modèle local
        local_model = fl_model.MyModel(input_shape, nbclasses=num_classes)
        local_model.set_weights(central_weights)
        
        # Entraîner le modèle local
        local_model.fit_it(trains=client_data, epochs=edge_epochs, tests=test_data, verbose=verbose)
        
        # Calculer le facteur d'échelle
        scaling_factor = _weight_scaling_factor(edges, client_name)
        scaled_weights = _scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        
        # Nettoyer la session
        K.clear_session()
    
    # Agréger les poids et mettre à jour le modèle central
    updated_weights = aggregation_fn(scaled_local_weight_list)
    central_model.set_weights(updated_weights)
    
    return central_model
```

<div class="page"/>

Cette fonction implémente les étapes clés de l'apprentissage fédéré horizontal :
1. Distribution du modèle global à tous les clients
2. Entraînement local sur chaque client pendant un nombre spécifié d'époques
3. Calcul d'un facteur d'échelle pour chaque client proportionnel à la quantité de données dont il dispose
4. Agrégation des poids mis à l'échelle via une fonction d'agrégation spécifiée
5. Mise à jour du modèle central avec les poids agrégés

### Distribution des données entre clients

Un aspect essentiel de notre étude était d'expérimenter avec différentes stratégies de distribution des données entre clients. Nous avons implémenté trois types de distribution :

1. **Distribution IID** (Independent and Identically Distributed) : les données sont réparties aléatoirement entre les clients, garantissant une distribution similaire des classes pour chaque client.

```python
def iid_partition(X, y, num_clients, batch_size=32):
    '''Distribution IID: chaque client a des données indépendantes et identiquement distribuées'''
    data = list(zip(X, y))
    random.shuffle(data)
    
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]
    
    client_data = {}
    for i in range(num_clients):
        client_name = f'edge_{i}'
        
        client_X, client_y = zip(*shards[i])
        client_X, client_y = list(client_X), list(client_y)
        
        dataset = tf.data.Dataset.from_tensor_slices((client_X, client_y))
        dataset = dataset.shuffle(len(client_y))
        dataset = dataset.batch(batch_size)
        
        client_data[client_name] = dataset
    
    return client_data
```

<div class="page"/>

2. **Distribution non-IID** : chaque client reçoit un sous-ensemble biaisé des classes, créant une hétérogénéité dans la distribution des données.

```python
def non_iid_partition(X, y, num_clients, classes_per_client=2, batch_size=32):
    '''Distribution non-IID: chaque client a un sous-ensemble déséquilibré des classes'''
    labels = np.argmax(y, axis=1)
    sorted_indices = np.argsort(labels)
    
    X_sorted = [X[i] for i in sorted_indices]
    y_sorted = [y[i] for i in sorted_indices]
    
    # Compter combien de classes nous avons
    num_classes = y[0].shape[0]
    samples_per_class = len(X) // num_classes
    
    # Répartir les données entre les clients
    client_data = {}
    for i in range(num_clients):
        client_name = f'edge_{i}'
        client_X, client_y = [], []
        
        # Pour chaque client, sélectionner classes_per_client classes
        selected_classes = np.random.choice(range(num_classes), classes_per_client, replace=False)
        
        # ... [code pour allouer les données selon les classes sélectionnées] ...
        
        # Créer le dataset pour ce client
        dataset = tf.data.Dataset.from_tensor_slices((client_X, client_y))
        dataset = dataset.shuffle(len(client_y))
        dataset = dataset.batch(batch_size)
        
        client_data[client_name] = dataset
    
    return client_data
```

3. **Distribution non-IID extrême** : chaque client se spécialise presque exclusivement sur une ou deux classes, avec très peu d'exemples des autres classes.

Ces différentes distributions nous permettent d'étudier la robustesse des algorithmes d'apprentissage fédéré face à l'hétérogénéité des données, un défi majeur dans les applications réelles.

<div class="page"/>

### Algorithmes d'agrégation

Nous avons implémenté et comparé trois algorithmes d'agrégation principaux :

1. **FedAvg** (Federated Averaging) : l'algorithme standard qui calcule une moyenne pondérée des poids des modèles locaux.

```python
def fedavg(scaled_weight_list, central_weights=None, config_name=None, round_num=None): 
    avg_weights = list()
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_weights.append(layer_mean)
    
    return avg_weights
```

2. **FedSGD** (Federated Stochastic Gradient Descent) : agrégation basée sur les gradients plutôt que sur les poids eux-mêmes.

```python
def fedsgd(model, client_grads, learning_rate=0.01, central_weights=None, config_name=None, round_num=None):
    current_weights = model.get_weights()
    updated_weights = []
    for i in range(len(current_weights)):
        updated_weights.append(current_weights[i] - learning_rate * avg_grads[i])
    
    return updated_weights
```

3. **FedProx** (Federated Proximal) : une extension de FedAvg qui ajoute un terme de régularisation proximal pour limiter la divergence entre les modèles.

```python
def fedprox(scaled_weight_list, global_weights, mu=0.01, config_name=None, round_num=None):
    avg_weights = []
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_weights.append(layer_mean) 
    for i in range(len(avg_weights)):
        proximal_term = mu * (avg_weights[i] - global_weights[i])
        avg_weights[i] = avg_weights[i] - proximal_term
    
    return avg_weights
```

## Paramètres étudiés et expérimentations

Dans nos expériences, nous avons fait varier plusieurs paramètres clés :

1. **Nombre de rounds fédérés** (3, 5, 10) : combien de fois les clients et le serveur échangent des mises à jour.
2. **Nombre de clients** (5, 10, 20) : combien d'entités participent à l'apprentissage fédéré.
3. **Distribution des données** (IID, non-IID, non-IID extrême) : comment les données sont réparties entre les clients.
4. **Nombre d'époques locales** (1, 3, 5) : combien d'époques d'entraînement chaque client effectue localement.
5. **Algorithmes d'agrégation** (FedAvg, FedSGD avec différents taux d'apprentissage, FedProx avec différentes valeurs du paramètre μ).

Ces expériences ont été exécutées via le script `run_all.py`, qui implémente une boucle complète d'expérimentation :

```python
def run_all_experiments():
    # ... [initialisation] ...
    
    configurations = [
        {'name': 'Rounds_3', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3, 'rounds': 3},
        {'name': 'Rounds_5', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3, 'rounds': 5},
        # ... [autres configurations] ...
    ]
    
    for config in configurations:
        # ... [répartition des données selon la configuration] ...
        
        for round_num in range(num_rounds):
            federated_model = fl_types.horizontal_federated_learning(
                edges=edges,
                central_model=federated_model,
                input_shape=input_shape,
                num_classes=10,
                edge_epochs=config['epochs'],
                test_data=test_dataset,
                aggregation_fn=agg_fn,
                verbose=0
            )
            
            round_loss, round_acc = federated_model.evaluate(test_dataset, verbose=0)
            # ... [enregistrement des résultats] ...
```

## Résultats et analyse

### Comparaison avec l'entraînement centralisé

Notre modèle de référence, entraîné de manière centralisée avec accès à toutes les données, atteint une précision de 95% après 10 époques. Aucune des configurations fédérées n'atteint tout à fait cette performance, ce qui était attendu étant donné les contraintes de l'apprentissage distribué. Cependant, les meilleures configurations fédérées parviennent à des performances respectables, jusqu'à 92% dans certains cas.

<div align="center">
    <img src="./plots/global_accuracy.png" width="700" />
</div>

<div class="page"/>

### Influence du nombre de rounds fédérés

Nos expériences montrent que la précision augmente avec le nombre de rounds, mais avec des rendements décroissants. Après 5 rounds, les configurations avec 3, 5 et 10 rounds atteignent respectivement 90,7%, 90,5% et 90,1% de précision. Cette similitude suggère qu'au-delà de 5 rounds, le gain de performance devient marginal pour cette tâche.

<div align="center">
    <img src="./plots/convergence_by_rounds.png" width="500" />
</div>

La convergence suit une courbe typique d'apprentissage : rapide au début, puis qui ralentit progressivement. Cette observation est importante pour optimiser les ressources dans un déploiement réel.

### Impact du nombre de clients

Nous avons observé une relation inverse entre le nombre de clients et la précision du modèle final. Avec 5 clients, la précision atteint 92%, tandis qu'avec 10 et 20 clients, elle baisse respectivement à 90% et 87%.

<div align="center">
    <img src="./plots/clients_vs_time.png" width="500" />
</div>

<div class="page"/>

Cette dégradation peut s'expliquer par plusieurs facteurs :
1. Moins de données par client, limitant la capacité d'apprentissage individuelle
2. Plus grande diversité de mises à jour, créant potentiellement des interférences lors de l'agrégation
3. Augmentation quasi-linéaire du temps de calcul (125s pour 5 clients, 213s pour 10 clients, 408s pour 20 clients)

### Influence de la distribution des données

La distribution des données entre clients s'est avérée être le facteur le plus critique. En configuration IID, le modèle atteint 90% de précision. En revanche, en configuration non-IID, la performance chute drastiquement à 48,2%.

<div align="center">
    <img src="./plots/iid_vs_noniid_impact.png" width="600" />
</div>

Fait intéressant, dans le scénario non-IID extrême, la précision remonte à 62%. Cette observation contre-intuitive pourrait s'expliquer par le fait que chaque client devient "expert" dans la reconnaissance de classes spécifiques, compensant partiellement les défis de l'hétérogénéité.

<div class="page"/>

### Effet du nombre d'époques locales

Avec une seule époque locale, la précision finale atteint 85,7%. Ce chiffre augmente à 90,7% avec 3 époques et à 92,2% avec 5 époques. Cette progression montre qu'un plus grand nombre d'époques permet à chaque modèle local de mieux apprendre à partir de ses données.

<div align="center">
    <img src="./plots/epochs_vs_accuracy.png" width="500" />
</div>

<div align="center">
    <img src="./plots/local_epochs_effect.png" width="500" />
</div>

Toutefois, ce gain se fait au prix d'un temps de calcul plus élevé : 189s pour 1 époque, 249s pour 3 époques, et 295s pour 5 époques. Il existe également un risque qu'un trop grand nombre d'époques conduise à une sur-spécialisation des modèles sur leurs données locales.

<div class="page"/>

### Comparaison des algorithmes d'agrégation

FedAvg, avec une précision de 90%, offre un bon équilibre entre simplicité et performance. En revanche, FedSGD montre des performances très faibles, avec seulement 10,8% de précision pour des taux d'apprentissage de 0,01 et 0,1 respectivement.

![Précision vs Algorithme d'agrégation](./plots/algorithm_vs_accuracy.png)

FedProx affiche des résultats prometteurs avec 90% de précision pour des valeurs de μ de 0,01 et 0,1 respectivement. Ces résultats légèrement supérieurs à FedAvg suggèrent que le terme de régularisation proximal aide à stabiliser l'apprentissage.

Les visualisations de progression des paramètres montrent des différences significatives dans les trajectoires d'apprentissage de ces algorithmes:

![Progression FedProx](./progression_plots/FedProx_mu0.01_fedprox_progression.png)

<div class="page"/>

![Progression FedSGD](./progression_plots/FedSGD_LR0.01_fedsgd_progression.png)

FedAvg et FedProx convergent vers des régions similaires de l'espace des paramètres, tandis que FedSGD semble stagner.

<div class="page"/>

## Conclusion et perspectives

Notre étude a permis d'identifier plusieurs facteurs clés influençant l'efficacité de l'apprentissage fédéré. Si cette approche n'atteint pas les performances d'un modèle centralisé ayant accès à toutes les données, elle offre néanmoins un compromis intéressant entre confidentialité et précision.

Les défis majeurs concernent la gestion de l'hétérogénéité des données entre clients et l'équilibre optimal entre le nombre de rounds globaux et d'époques locales. L'algorithme d'agrégation joue également un rôle crucial, avec FedProx démontrant une légère supériorité sur FedAvg dans nos tests.

Ces observations fournissent des pistes pour l'optimisation des systèmes d'apprentissage fédéré dans des applications réelles, où les contraintes de ressources, de confidentialité et de performance doivent être équilibrées. Notre étude démontre qu'avec une configuration appropriée, l'apprentissage fédéré peut atteindre des performances compétitives tout en préservant la confidentialité des données.

Pour des développements futurs, il serait intéressant d'explorer des méthodes plus robustes pour gérer les données non-IID, des techniques de communication plus efficaces pour réduire la bande passante requise, ainsi que des approches de personnalisation adaptant le modèle global aux spécificités locales sans compromettre la performance globale.
