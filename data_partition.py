import numpy as np
import random
import tensorflow as tf

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
        
        for cls in selected_classes:
            # Calculer les indices de début et de fin pour cette classe
            start_idx = cls * samples_per_class
            end_idx = (cls + 1) * samples_per_class
            
            # Allouer samples_per_client/classes_per_client échantillons de chaque classe
            samples_per_client = max(1, samples_per_class // (num_clients // classes_per_client))
            
            # Calculer l'offset pour ce client dans cette classe
            offset = (i % (num_clients // classes_per_client)) * samples_per_client
            if offset + samples_per_client > samples_per_class:
                offset = 0
                
            # Ajouter les données pour ce client
            client_X.extend(X_sorted[start_idx + offset:start_idx + offset + samples_per_client])
            client_y.extend(y_sorted[start_idx + offset:start_idx + offset + samples_per_client])
        
        # Créer le dataset pour ce client
        dataset = tf.data.Dataset.from_tensor_slices((client_X, client_y))
        dataset = dataset.shuffle(len(client_y))
        dataset = dataset.batch(batch_size)
        
        client_data[client_name] = dataset
    
    return client_data

def non_iid_extreme_partition(X, y, num_clients, classes_per_client=1, batch_size=32):
    '''Distribution non-IID extrême: chaque client a principalement une seule classe'''
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
        
        # Chaque client prend surtout une classe dominante (80% des données) 
        # et quelques échantillons des autres classes (20%)
        dominant_class = i % num_classes
        
        # Calculer les indices de début et de fin pour la classe dominante
        start_idx = dominant_class * samples_per_class
        end_idx = (dominant_class + 1) * samples_per_class
        
        # Nombre d'échantillons de la classe dominante (80%)
        dominant_samples = int(samples_per_class * 0.8 / (num_clients // num_classes))
        
        # Offset pour ce client dans la classe dominante
        offset = (i // num_classes) * dominant_samples
        if offset + dominant_samples > samples_per_class:
            offset = 0
            
        # Ajouter les échantillons de la classe dominante
        client_X.extend(X_sorted[start_idx + offset:start_idx + offset + dominant_samples])
        client_y.extend(y_sorted[start_idx + offset:start_idx + offset + dominant_samples])
        
        # Ajouter quelques échantillons des autres classes (20%)
        other_samples_total = int(dominant_samples * 0.25)  # 20% de données supplémentaires
        other_samples_per_class = other_samples_total // (num_classes - 1)
        
        for cls in range(num_classes):
            if cls == dominant_class:
                continue
                
            # Calculer les indices pour cette classe
            cls_start_idx = cls * samples_per_class
            
            # Prendre quelques échantillons de cette classe
            random_offset = np.random.randint(0, samples_per_class - other_samples_per_class)
            client_X.extend(X_sorted[cls_start_idx + random_offset:cls_start_idx + random_offset + other_samples_per_class])
            client_y.extend(y_sorted[cls_start_idx + random_offset:cls_start_idx + random_offset + other_samples_per_class])
        
        # Mélanger les données du client
        combined = list(zip(client_X, client_y))
        random.shuffle(combined)
        client_X, client_y = zip(*combined)
        
        # Créer le dataset pour ce client
        dataset = tf.data.Dataset.from_tensor_slices((client_X, client_y))
        dataset = dataset.shuffle(len(client_y))
        dataset = dataset.batch(batch_size)
        
        client_data[client_name] = dataset
    
    return client_data