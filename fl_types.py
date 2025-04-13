import tensorflow as tf
import numpy as np
import random
from tensorflow.keras import backend as K
import fl_model

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

def horizontal_federated_learning_with_proximal(edges, central_model, input_shape, num_classes, 
                                 edge_epochs, test_data, aggregation_fn, mu=0.01, verbose=0):
    '''Apprentissage fédéré horizontal avec terme proximal (FedProx)'''
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
        
        # Modifier la fonction de perte pour inclure le terme proximal
        original_loss = local_model.loss_fn
        proximal_term = lambda model, w0=central_weights, mu=mu: mu * sum(
            tf.reduce_sum(tf.square(w - w0_i)) 
            for w, w0_i in zip(model.get_weights(), w0)
        )
        
        # Entraîner le modèle local avec la régularisation proximale
        # Note: Ceci est conceptuel, l'implémentation exacte dépendra de la façon
        # dont la fonction d'entraînement est configurée dans fl_model.py
        local_model.fit_it_with_proximal(
            trains=client_data, 
            epochs=edge_epochs, 
            tests=test_data, 
            verbose=verbose, 
            global_weights=central_weights,
            mu=mu
        )
        
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

def _weight_scaling_factor(edges, edge_name):
    '''Calculer le facteur d'échelle: n_k/n'''
    all_edge_names = list(edges.keys())
    
    # Calculer le nombre total d'exemples
    batch_size = next(iter(edges[edge_name]))[0].shape[0]
    global_count = sum([tf.data.experimental.cardinality(edges[name]).numpy() for name in all_edge_names]) * batch_size
    
    # Calculer le nombre d'exemples pour ce client
    local_count = tf.data.experimental.cardinality(edges[edge_name]).numpy() * batch_size
    
    return local_count / global_count

def _scale_model_weights(weights, scalar):
    '''Mettre à l'échelle les poids du modèle par un scalaire'''
    scaled_weights = []
    for weight in weights:
        scaled_weights.append(scalar * weight)
    return scaled_weights