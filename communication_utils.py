import numpy as np
import tensorflow as tf

def count_parameters(model_weights):
    """
    Compte le nombre de paramètres dans un modèle
    
    Args:
        model_weights: Liste des poids du modèle
        
    Returns:
        int: Nombre total de paramètres
    """
    total_params = 0
    for w in model_weights:
        total_params += np.prod(w.shape)
    return int(total_params)

def calculate_weights_size(model_weights):
    """
    Calcule la taille des poids du modèle en octets
    
    Args:
        model_weights: Liste des poids du modèle
        
    Returns:
        int: Taille totale en octets
    """
    total_size = 0
    for w in model_weights:
        # Chaque nombre flottant est généralement stocké sur 4 octets (float32)
        total_size += np.prod(w.shape) * 4
    return total_size

def track_communication(func):
    """
    Décorateur pour suivre les communications dans les fonctions d'apprentissage fédéré
    """
    def wrapper(*args, **kwargs):
        # Accéder aux paramètres pertinents
        edges = args[0]
        central_model = args[1]
        
        # Calculer la taille des poids du modèle central
        central_weights = central_model.get_weights()
        central_weights_size = calculate_weights_size(central_weights)
        
        # Nombre de clients
        num_clients = len(edges)
        
        # Taille totale des communications descendantes (du serveur vers les clients)
        downstream_size = central_weights_size * num_clients
        
        # Exécuter la fonction d'origine
        result = func(*args, **kwargs)
        
        # Calculer la taille des communications montantes (des clients vers le serveur)
        # Nous supposons que tous les clients envoient leurs poids mis à l'échelle
        upstream_size = central_weights_size * num_clients
        
        # Taille totale des communications
        total_communication = downstream_size + upstream_size
        
        # Stocker les informations de communication dans un registre global
        # (à implémenter dans la fonction principale)
        communication_stats = {
            'num_clients': num_clients,
            'num_parameters': count_parameters(central_weights),
            'downstream_size': downstream_size,
            'upstream_size': upstream_size,
            'total_size': total_communication
        }
        
        # Accéder au contexte ou à la variable globale pour stocker les statistiques
        if 'config_name' in kwargs:
            communication_stats['config_name'] = kwargs['config_name']
        if 'round_num' in kwargs:
            communication_stats['round_num'] = kwargs['round_num']
        
        # Nous retournons les statistiques avec le résultat original
        return result, communication_stats
    
    return wrapper

class CommunicationTracker:
    """
    Classe pour suivre les communications dans l'apprentissage fédéré
    """
    def __init__(self):
        self.stats = []
    
    def record_communication(self, config_name, round_num, num_clients, num_parameters, downstream_size, upstream_size):
        """
        Enregistre les statistiques de communication pour un round donné
        """
        self.stats.append({
            'config_name': config_name,
            'round': round_num,
            'num_clients': num_clients,
            'num_parameters': num_parameters,
            'downstream_size': downstream_size,
            'upstream_size': upstream_size,
            'total_size': downstream_size + upstream_size
        })
    
    def get_stats(self):
        """
        Retourne toutes les statistiques de communication
        """
        return self.stats