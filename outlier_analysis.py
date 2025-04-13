import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fl_model
import fl_types
import aggregation
from fairness_utils import add_noise_to_client_data

def run_outlier_experiments(X_train, y_train, X_test, y_test, input_shape, 
                           num_clients=10, epochs=3, rounds=5, noise_levels=[0.1, 0.3, 0.5], 
                           outlier_ratios=[0.1, 0.2, 0.3], batch_size=32):
    """
    Exécute des expériences avec différents niveaux de bruit et proportions de clients outliers
    """
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    
    results = []
    
    # Pour chaque niveau de bruit
    for noise_level in noise_levels:
        # Pour chaque proportion de clients outliers
        for outlier_ratio in outlier_ratios:
            print(f"Expérience avec bruit={noise_level}, ratio d'outliers={outlier_ratio}")
            
            # Distribution IID standard
            edges = {}
            client_datasets = []
            
            # Nombre de clients outliers
            num_outlier_clients = int(num_clients * outlier_ratio)
            
            # Assigner les données à tous les clients (IID)
            data = list(zip(X_train, y_train))
            np.random.shuffle(data)
            chunk_size = len(data) // num_clients
            
            # Créer les datasets pour chaque client
            for i in range(num_clients):
                client_name = f'edge_{i}'
                
                # Sélectionner un chunk de données pour ce client
                client_data = data[i * chunk_size:(i + 1) * chunk_size]
                client_X, client_y = zip(*client_data)
                client_X, client_y = list(client_X), list(client_y)
                
                # Créer un dataset
                client_dataset = tf.data.Dataset.from_tensor_slices((client_X, client_y))
                client_dataset = client_dataset.shuffle(len(client_y))
                client_dataset = client_dataset.batch(batch_size)
                
                # Stocker le dataset client
                client_datasets.append(client_dataset)
            
            # Ajouter du bruit aux clients outliers
            outlier_indices = np.random.choice(num_clients, num_outlier_clients, replace=False)
            for i, dataset in enumerate(client_datasets):
                if i in outlier_indices:
                    # Client outlier - ajouter du bruit
                    client_datasets[i] = add_noise_to_client_data(dataset, noise_level)
            
            # Créer le dictionnaire final des edges
            for i in range(num_clients):
                edges[f'edge_{i}'] = client_datasets[i]
            
            # Créer et configurer le modèle fédéré
            federated_model = fl_model.MyModel(input_shape, nbclasses=10)
            
            # Historique des métriques
            round_history = {'round': [], 'loss': [], 'accuracy': []}
            
            # Exécuter l'apprentissage fédéré
            for round_num in range(rounds):
                print(f"  Round {round_num+1}/{rounds}")
                
                # Définir la fonction d'agrégation (FedAvg)
                agg_fn = lambda x: aggregation.fedavg(x)
                
                # Exécuter un round d'apprentissage fédéré
                federated_model = fl_types.horizontal_federated_learning(
                    edges=edges,
                    central_model=federated_model,
                    input_shape=input_shape,
                    num_classes=10,
                    edge_epochs=epochs,
                    test_data=test_dataset,
                    aggregation_fn=agg_fn,
                    verbose=0
                )
                
                # Évaluer après chaque round
                round_loss, round_acc = federated_model.evaluate(test_dataset, verbose=0)
                
                # Enregistrer les métriques du round
                round_history['round'].append(round_num + 1)
                round_history['loss'].append(round_loss)
                round_history['accuracy'].append(round_acc)
                
                print(f"    Loss: {round_loss:.4f}, Accuracy: {round_acc:.4f}")
            
            # Évaluer le modèle fédéré final
            final_loss, final_acc = federated_model.evaluate(test_dataset, verbose=0)
            
            # Enregistrer les résultats
            results.append({
                'noise_level': noise_level,
                'outlier_ratio': outlier_ratio,
                'final_loss': final_loss,
                'final_accuracy': final_acc,
                'history': round_history
            })
            
            # Libérer la mémoire
            tf.keras.backend.clear_session()
    
    return results

def visualize_outlier_results(results, save_dir='outlier_plots'):
    """
    Visualise les résultats des expériences avec outliers
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Impact du bruit sur la précision finale
    plt.figure(figsize=(12, 8))
    for outlier_ratio in sorted(set(r['outlier_ratio'] for r in results)):
        # Filtrer les résultats pour ce ratio
        filtered_results = [r for r in results if r['outlier_ratio'] == outlier_ratio]
        # Trier par niveau de bruit
        filtered_results.sort(key=lambda x: x['noise_level'])
        
        noise_levels = [r['noise_level'] for r in filtered_results]
        accuracies = [r['final_accuracy'] for r in filtered_results]
        
        plt.plot(noise_levels, accuracies, marker='o', label=f'Outliers: {outlier_ratio*100}%')
    
    plt.title('Impact du niveau de bruit sur la précision')
    plt.xlabel('Niveau de bruit')
    plt.ylabel('Précision finale')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.savefig(f'{save_dir}/noise_level_impact.png')
    plt.close()
    
    # 2. Impact du ratio d'outliers sur la précision finale
    plt.figure(figsize=(12, 8))
    for noise_level in sorted(set(r['noise_level'] for r in results)):
        # Filtrer les résultats pour ce niveau de bruit
        filtered_results = [r for r in results if r['noise_level'] == noise_level]
        # Trier par ratio d'outliers
        filtered_results.sort(key=lambda x: x['outlier_ratio'])
        
        outlier_ratios = [r['outlier_ratio'] for r in filtered_results]
        accuracies = [r['final_accuracy'] for r in filtered_results]
        
        plt.plot(outlier_ratios, accuracies, marker='o', label=f'Bruit: {noise_level}')
    
    plt.title("Impact du ratio d'outliers sur la précision")
    plt.xlabel("Ratio d'outliers")
    plt.ylabel('Précision finale')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.savefig(f'{save_dir}/outlier_ratio_impact.png')
    plt.close()
    
    # 3. Heatmap précision vs (bruit, ratio d'outliers)
    noise_levels = sorted(set(r['noise_level'] for r in results))
    outlier_ratios = sorted(set(r['outlier_ratio'] for r in results))
    
    heatmap_data = np.zeros((len(noise_levels), len(outlier_ratios)))
    
    for i, noise in enumerate(noise_levels):
        for j, ratio in enumerate(outlier_ratios):
            # Trouver le résultat correspondant
            for r in results:
                if r['noise_level'] == noise and r['outlier_ratio'] == ratio:
                    heatmap_data[i, j] = r['final_accuracy']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=[f'{r*100}%' for r in outlier_ratios],
                yticklabels=[f'{n}' for n in noise_levels])
    plt.title('Précision finale en fonction du bruit et du ratio d\'outliers')
    plt.xlabel('Ratio d\'outliers')
    plt.ylabel('Niveau de bruit')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/heatmap_accuracy.png')
    plt.close()
    
    # 4. Courbes de convergence pour différentes configurations
    plt.figure(figsize=(14, 10))
    
    # Sélectionner quelques configurations représentatives
    selected_configs = []
    
    # Ajouter les configurations avec bruit minimum et maximum
    min_noise = min(noise_levels)
    max_noise = max(noise_levels)
    mid_ratio = outlier_ratios[len(outlier_ratios)//2]
    
    for r in results:
        if (r['noise_level'] == min_noise and r['outlier_ratio'] == mid_ratio) or \
           (r['noise_level'] == max_noise and r['outlier_ratio'] == mid_ratio):
            selected_configs.append(r)
    
    # Ajouter les configurations avec ratio minimum et maximum
    min_ratio = min(outlier_ratios)
    max_ratio = max(outlier_ratios)
    mid_noise = noise_levels[len(noise_levels)//2]
    
    for r in results:
        if (r['outlier_ratio'] == min_ratio and r['noise_level'] == mid_noise) or \
           (r['outlier_ratio'] == max_ratio and r['noise_level'] == mid_noise):
            if r not in selected_configs:
                selected_configs.append(r)
    
    # Tracer les courbes de convergence
    for config in selected_configs:
        label = f"Bruit: {config['noise_level']}, Outliers: {config['outlier_ratio']*100}%"
        plt.plot(config['history']['round'], config['history']['accuracy'], marker='o', label=label)
    
    plt.title('Convergence avec différentes configurations d\'outliers')
    plt.xlabel('Round')
    plt.ylabel('Précision')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.savefig(f'{save_dir}/convergence_comparison.png')
    plt.close()