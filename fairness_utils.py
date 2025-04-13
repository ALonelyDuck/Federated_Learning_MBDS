import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_client_models(edges, input_shape, num_classes, central_weights, test_data):
    """
    Évalue les performances de chaque modèle client sur l'ensemble de test
    
    Args:
        edges: Dictionnaire contenant les données de chaque client
        input_shape: Forme des données d'entrée
        num_classes: Nombre de classes
        central_weights: Poids du modèle central à copier pour initialiser les modèles clients
        test_data: Données de test pour l'évaluation
        
    Returns:
        dict: Dictionnaire contenant les métriques pour chaque client
    """
    import fl_model
    
    client_metrics = {}
    
    for client_name, client_data in edges.items():
        # Créer un modèle pour ce client
        client_model = fl_model.MyModel(input_shape, nbclasses=num_classes)
        client_model.set_weights(central_weights)
        
        # Entraîner le modèle sur les données du client
        client_model.fit_it(trains=client_data, epochs=1, tests=test_data, verbose=0)
        
        # Évaluer le modèle sur l'ensemble de test
        loss, accuracy = client_model.evaluate(test_data, verbose=0)
        
        # Stocker les métriques
        client_metrics[client_name] = {
            'loss': loss,
            'accuracy': accuracy
        }
        
        # Libérer les ressources
        del client_model
        tf.keras.backend.clear_session()
    
    return client_metrics

def calculate_performance_variance(client_metrics):
    """
    Calcule la variance des performances entre les clients
    
    Args:
        client_metrics: Dictionnaire contenant les métriques pour chaque client
        
    Returns:
        dict: Dictionnaire contenant les statistiques sur les performances
    """
    accuracies = [metrics['accuracy'] for metrics in client_metrics.values()]
    losses = [metrics['loss'] for metrics in client_metrics.values()]
    
    stats = {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'accuracy_min': np.min(accuracies),
        'accuracy_max': np.max(accuracies),
        'loss_mean': np.mean(losses),
        'loss_std': np.std(losses),
        'loss_min': np.min(losses),
        'loss_max': np.max(losses)
    }
    
    return stats

def compute_client_confusion_matrices(edges, input_shape, num_classes, central_weights, test_data):
    """
    Calcule les matrices de confusion pour chaque client
    
    Args:
        edges: Dictionnaire contenant les données de chaque client
        input_shape: Forme des données d'entrée
        num_classes: Nombre de classes
        central_weights: Poids du modèle central à copier pour initialiser les modèles clients
        test_data: Données de test pour l'évaluation
        
    Returns:
        dict: Dictionnaire contenant la matrice de confusion pour chaque client
    """
    import fl_model
    
    confusion_matrices = {}
    
    # Extraire les étiquettes de test
    y_test = np.concatenate([y for _, y in test_data], axis=0)
    y_test_labels = np.argmax(y_test, axis=1)
    
    for client_name, client_data in edges.items():
        # Créer un modèle pour ce client
        client_model = fl_model.MyModel(input_shape, nbclasses=num_classes)
        client_model.set_weights(central_weights)
        
        # Entraîner le modèle sur les données du client
        client_model.fit_it(trains=client_data, epochs=1, tests=test_data, verbose=0)
        
        # Faire des prédictions sur l'ensemble de test
        x_test = np.concatenate([x for x, _ in test_data], axis=0)
        y_pred = client_model.model.predict(x_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        # Calculer la matrice de confusion
        cm = confusion_matrix(y_test_labels, y_pred_labels)
        
        # Stocker la matrice de confusion
        confusion_matrices[client_name] = cm
        
        # Libérer les ressources
        del client_model
        tf.keras.backend.clear_session()
    
    return confusion_matrices

def plot_client_confusion_matrices(confusion_matrices, save_dir='confusion_matrices'):
    """
    Visualise les matrices de confusion pour chaque client
    
    Args:
        confusion_matrices: Dictionnaire contenant la matrice de confusion pour chaque client
        save_dir: Répertoire où sauvegarder les visualisations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for client_name, cm in confusion_matrices.items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matrice de confusion pour {client_name}')
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{client_name}_confusion_matrix.png')
        plt.close()

def add_noise_to_client_data(client_data, noise_level=0.2):
    """
    Ajoute du bruit aux données d'un client
    
    Args:
        client_data: Données du client (tf.data.Dataset)
        noise_level: Niveau de bruit à ajouter (écart-type du bruit gaussien)
        
    Returns:
        tf.data.Dataset: Données bruitées
    """
    noisy_datasets = []
    
    # Parcourir les batchs du dataset
    for x, y in client_data:
        # Convertir en numpy pour manipulation
        x_np = x.numpy()
        
        # Ajouter du bruit gaussien
        noise = np.random.normal(0, noise_level, x_np.shape)
        x_noisy = x_np + noise
        
        # Limiter les valeurs entre 0 et 1 (pour les images)
        x_noisy = np.clip(x_noisy, 0, 1)
        
        # Créer un nouveau dataset pour ce batch
        batch_dataset = tf.data.Dataset.from_tensor_slices((x_noisy, y.numpy()))
        noisy_datasets.append(batch_dataset)
    
    # Concaténer tous les batchs bruités
    if noisy_datasets:
        noisy_dataset = noisy_datasets[0]
        for ds in noisy_datasets[1:]:
            noisy_dataset = noisy_dataset.concatenate(ds)
        
        # Rebatcher le dataset
        batch_size = next(iter(client_data))[0].shape[0]
        noisy_dataset = noisy_dataset.batch(batch_size)
        
        return noisy_dataset
    else:
        return client_data  # Retourner le dataset original si vide

def create_noisy_client_distribution(X_train, y_train, num_clients, noisy_client_ratio=0.2, noise_level=0.2, batch_size=32):
    """
    Crée une distribution de clients où certains ont des données bruitées
    """
    import data_partition
    
    # Distribution IID standard
    edges = data_partition.iid_partition(X_train, y_train, num_clients, batch_size)
    
    # Nombre de clients à bruiter
    num_noisy_clients = int(num_clients * noisy_client_ratio)
    
    # Sélectionner aléatoirement les clients à bruiter
    noisy_client_indices = np.random.choice(num_clients, num_noisy_clients, replace=False)
    noisy_client_names = [f'edge_{i}' for i in noisy_client_indices]
    
    # Ajouter du bruit aux données des clients sélectionnés
    for client_name in noisy_client_names:
        if client_name in edges:
            edges[client_name] = add_noise_to_client_data(edges[client_name], noise_level)
    
    return edges, noisy_client_names