import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from matplotlib.lines import Line2D
import pickle

# Dictionnaire global pour stocker les paramètres de chaque round pour la visualisation de progression
parameter_history = {}

def fedavg(scaled_weight_list, central_weights=None, config_name=None, round_num=None):
    '''FedAvg: Moyenne pondérée des poids'''    
    # Calculer la moyenne pondérée
    avg_weights = list()
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_weights.append(layer_mean)
    
    return avg_weights

def fedsgd(model, client_grads, learning_rate=0.01, central_weights=None, config_name=None, round_num=None):
    '''FedSGD: Agrégation des gradients plutôt que des poids'''
    # Calculer les gradients pour chaque client
    gradients_for_viz = []
    for weights in client_grads:
        # Calculer le gradient comme la différence entre les poids actuels et les poids du client
        grads = []
        for w_model, w_client in zip(model.get_weights(), weights):
            grads.append((w_model - w_client) / learning_rate)  # grad = -Δw/lr
        gradients_for_viz.append(grads)
    
    # Visualiser les paramètres et gradients
    if config_name and round_num is not None:
        visualize_with_gradients(client_grads, gradients_for_viz, central_weights, config_name, round_num, 'fedsgd')
        # Sauvegarder les paramètres pour la visualisation de progression
        save_parameters_for_progression(client_grads, central_weights, config_name, round_num, 'fedsgd')
    
    # Calculer la moyenne des gradients
    avg_grads = list()
    for grad_list_tuple in zip(*client_grads):
        layer_mean = tf.math.reduce_mean(grad_list_tuple, axis=0)
        avg_grads.append(layer_mean)
    
    # Calculer les normes des gradients
    grad_norms = [np.linalg.norm(g) for g in avg_grads if isinstance(g, np.ndarray)]
    if grad_norms:
        print(f"Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={np.mean(grad_norms):.6f}")
    
    # Mise à jour des poids avec les gradients moyens
    current_weights = model.get_weights()
    updated_weights = []
    for i in range(len(current_weights)):
        updated_weights.append(current_weights[i] - learning_rate * avg_grads[i])
    
    return updated_weights

def fedprox(scaled_weight_list, global_weights, mu=0.01, config_name=None, round_num=None):
    '''FedProx: FedAvg avec terme de régularisation proximal'''
    
    # Calculer la moyenne comme FedAvg
    avg_weights = []
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_weights.append(layer_mean)
    
    # Visualiser les paramètres si demandé
    if config_name and round_num is not None:
        save_parameters_for_progression(scaled_weight_list, global_weights, config_name, round_num, 'fedprox')
    
    return avg_weights

def save_parameters_for_progression(weights_list, central_weights, config_name, round_num, algo_type):
    """Sauvegarde les paramètres pour une visualisation ultérieure de la progression"""
    key = f"{config_name}_{algo_type}"
    if key not in parameter_history:
        parameter_history[key] = {}
    
    parameter_history[key][round_num] = {
        'weights': weights_list,
        'central': central_weights
    }
    
    # Sauvegarder dans un fichier pour persistance
    os.makedirs('parameter_history', exist_ok=True)
    with open(f'parameter_history/{key}.pkl', 'wb') as f:
        pickle.dump(parameter_history[key], f)

def visualize_with_gradients(weights_list, gradients_list, central_weights, config_name, round_num, algo_type):
    """Visualisation améliorée avec gradients sous forme de flèches"""
    plots_dir = 'gradient_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Extraire les poids
    flattened_weights = []
    client_ids = []
    layer_idx = 1
    
    for client_idx, client_weights in enumerate(weights_list):
        if len(client_weights) > layer_idx:
            param = client_weights[layer_idx]
            if isinstance(param, np.ndarray) and param.size > 0:
                flattened_weights.append(param.flatten())
                client_ids.append(client_idx)
    
    # Ajouter les poids centraux
    if central_weights is not None and len(central_weights) > layer_idx:
        central_param = central_weights[layer_idx]
        if isinstance(central_param, np.ndarray) and central_param.size > 0:
            flattened_weights.append(central_param.flatten())
            client_ids.append(-1)
    
    # Extraire les gradients correspondants
    flattened_gradients = []
    for client_idx, client_grads in enumerate(gradients_list):
        if client_idx < len(client_ids) and client_ids[client_idx] != -1:  # Exclure le modèle central
            if len(client_grads) > layer_idx:
                grad = client_grads[layer_idx]
                if isinstance(grad, np.ndarray) and grad.size > 0:
                    flattened_gradients.append(grad.flatten())
    
    # S'il n'y a pas assez de données, sortir
    if len(flattened_weights) < 2:
        print(f"Pas assez de paramètres à visualiser pour {algo_type}")
        return
    
    # Convertir en tableaux numpy
    X = np.array(flattened_weights)
    
    # Réduction de dimensionnalité avec PCA
    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(X)
    
    # Si nous avons des gradients, les transformer aussi
    G_pca_2d = []
    if flattened_gradients:
        G = np.array(flattened_gradients)
        # Projeter les gradients dans le même espace PCA
        for grad in G:
            # Appliquer la même transformation
            g_transformed = pca.transform(grad.reshape(1, -1)).flatten()
            G_pca_2d.append(g_transformed)
        G_pca_2d = np.array(G_pca_2d)
    
    # Création de la palette de couleurs
    colors = ['red' if id == -1 else plt.cm.viridis(id/max(1, max(client_ids))) for id in client_ids]
    
    # Visualisation 2D avec PCA et gradients
    plt.figure(figsize=(14, 12))
    
    # Tracer les points des clients
    for i, (x, y) in enumerate(X_pca_2d):
        marker = 'X' if client_ids[i] == -1 else 'o'
        size = 200 if client_ids[i] == -1 else 100
        label = 'Modèle central' if client_ids[i] == -1 else f'Client {client_ids[i]}'
        plt.scatter(x, y, c=[colors[i]], marker=marker, s=size, label=label if client_ids[i] == -1 or client_ids[i] < 3 else None)
    
    # Tracer les gradients comme des flèches si disponibles
    if flattened_gradients and len(G_pca_2d) > 0:
        # Normaliser les gradients pour une meilleure visualisation
        grad_norms = np.sqrt(np.sum(G_pca_2d**2, axis=1))
        scale_factor = 0.01  # Petit facteur d'échelle pour éviter des flèches trop grandes
        
        for i, (x, y) in enumerate(X_pca_2d[:-1]):  # Exclure le modèle central
            if i < len(G_pca_2d):
                # Normaliser le gradient pour qu'il ait une longueur constante
                norm = grad_norms[i]
                if norm > 0:
                    dx = -G_pca_2d[i][0] / norm * scale_factor
                    dy = -G_pca_2d[i][1] / norm * scale_factor
                    
                    plt.arrow(x, y, dx, dy,
                             head_width=0.02, head_length=0.03, fc=colors[i], ec=colors[i])
    
    plt.title(f'Visualisation PCA 2D avec gradients {algo_type}\n{config_name} - Round {round_num}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(f'{plots_dir}/{config_name}_{algo_type}_round{round_num}_gradients_arrows.png')
    plt.close()
    
    print(f"Visualisation avec flèches de gradient enregistrée dans {plots_dir}/")

def generate_progression_visualizations():
    """Génère des visualisations de progression pour tous les algorithmes et configurations enregistrés"""
    # Parcourir tous les fichiers dans parameter_history
    history_dir = 'parameter_history'
    if not os.path.exists(history_dir):
        print("Aucun historique de paramètres trouvé")
        return
    
    for filename in os.listdir(history_dir):
        if filename.endswith('.pkl'):
            # Charger l'historique
            with open(f'{history_dir}/{filename}', 'rb') as f:
                history = pickle.load(f)
            
            # Extraire le nom de config et l'algo à partir du nom de fichier
            parts = filename.split('_')
            algo_type = parts[-1].replace('.pkl', '')
            config_name = '_'.join(parts[:-1])
            
            # Générer la visualisation de progression
            visualize_training_progression(history, config_name, algo_type)

def visualize_training_progression(history, config_name, algo_type):
    """Combine les visualisations de plusieurs rounds pour montrer la progression"""
    plots_dir = 'progression_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Obtenir le nombre de rounds
    max_rounds = max(history.keys())
    
    # Préparer la figure
    plt.figure(figsize=(16, 14))
    
    # Couleurs pour les rounds (du plus clair au plus foncé)
    round_colors = plt.cm.plasma(np.linspace(0.8, 0.2, max_rounds))
    
    # Pour chaque round, extraire les positions clients et central
    all_positions = {}  # {round: {client_id: position_2d}}
    all_centrals = {}   # {round: position_2d}
    
    # Premier passage pour obtenir toutes les positions dans le même espace PCA
    all_weights = []
    all_client_ids = []
    round_markers = []
    
    layer_idx = 1  # Indice de la couche à visualiser
    
    # Collecter tous les poids de tous les rounds
    for round_num in sorted(history.keys()):
        round_data = history[round_num]
        
        # Extraire et aplatir les poids clients
        for client_idx, client_weights in enumerate(round_data['weights']):
            if len(client_weights) > layer_idx:
                param = client_weights[layer_idx]
                if isinstance(param, np.ndarray) and param.size > 0:
                    all_weights.append(param.flatten())
                    all_client_ids.append(client_idx)
                    round_markers.append(round_num)
        
        # Extraire et aplatir les poids centraux
        if round_data['central'] is not None and len(round_data['central']) > layer_idx:
            central_param = round_data['central'][layer_idx]
            if isinstance(central_param, np.ndarray) and central_param.size > 0:
                all_weights.append(central_param.flatten())
                all_client_ids.append(-1)  # ID spécial pour le modèle central
                round_markers.append(round_num)
    
    if len(all_weights) < 2:
        print(f"Pas assez de données pour générer la visualisation de progression pour {config_name}_{algo_type}")
        return
    
    # Appliquer PCA à toutes les données combinées
    pca = PCA(n_components=2)
    all_positions_2d = pca.fit_transform(np.array(all_weights))
    
    # Organiser les positions par round et client_id
    idx = 0
    for round_num, client_id, pos_2d in zip(round_markers, all_client_ids, all_positions_2d):
        if round_num not in all_positions:
            all_positions[round_num] = {}
        if client_id == -1:
            all_centrals[round_num] = pos_2d
        else:
            all_positions[round_num][client_id] = pos_2d
        idx += 1
    
    # Tracer les points pour chaque round
    for round_num in sorted(all_positions.keys()):
        # Tracer les clients pour ce round
        for client_id, pos in all_positions[round_num].items():
            alpha = 0.3 + 0.7 * (round_num / max_rounds)  # Transparence augmente avec les rounds
            size = 80
            
            # Mettre des labels uniquement pour le dernier round
            label = None
            if round_num == max_rounds:
                label = f'Client {client_id}'
                if client_id > 2:  # Limiter le nombre de labels clients
                    label = None
            
            plt.scatter(pos[0], pos[1], c=[round_colors[round_num-1]], 
                       marker='o', s=size, alpha=alpha, label=label,
                       edgecolors='black' if client_id == 0 else None)
        
        # Tracer le modèle central pour ce round
        if round_num in all_centrals:
            central_pos = all_centrals[round_num]
            plt.scatter(central_pos[0], central_pos[1], c=[round_colors[round_num-1]], 
                       marker='X', s=150, alpha=1.0, 
                       label='Modèle central' if round_num == max_rounds else None,
                       edgecolors='black')
            
            # Ajouter le numéro du round
            plt.annotate(f"R{round_num}", (central_pos[0], central_pos[1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
    
    # Tracer les lignes entre rounds consécutifs pour le modèle central
    central_xs, central_ys = [], []
    for round_num in sorted(all_centrals.keys()):
        pos = all_centrals[round_num]
        central_xs.append(pos[0])
        central_ys.append(pos[1])
    
    plt.plot(central_xs, central_ys, 'r--', alpha=0.6, linewidth=1.5)
    
    # Tracer des flèches de progression pour les clients
    for client_id in set(id for round_positions in all_positions.values() for id in round_positions.keys()):
        positions = []
        for round_num in sorted(all_positions.keys()):
            if client_id in all_positions[round_num]:
                positions.append(all_positions[round_num][client_id])
        
        if len(positions) >= 2:
            # Tracer des lignes de progression
            client_xs = [pos[0] for pos in positions]
            client_ys = [pos[1] for pos in positions]
            plt.plot(client_xs, client_ys, '--', alpha=0.3, linewidth=0.5)
    
    plt.title(f'Progression de l\'entraînement {algo_type} - {config_name}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(alpha=0.3)
    
    # Ajouter une légende de couleur pour les rounds
    legend_elements = []
    for r in range(1, max_rounds+1):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=round_colors[r-1], markersize=10,
                                     label=f'Round {r}'))
    
    # Ajouter la légende des rounds
    first_legend = plt.legend(handles=legend_elements, title="Progression", 
                           loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.gca().add_artist(first_legend)
    
    # Ajouter la légende des clients
    plt.legend(title="Clients", loc='upper left', bbox_to_anchor=(1.01, 0.6))
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/{config_name}_{algo_type}_progression.png')
    plt.close()
    
    print(f"Visualisation de progression enregistrée dans {plots_dir}/")