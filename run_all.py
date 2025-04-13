import fl_dataquest
import fl_model
import data_partition
import aggregation
import fl_types
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
from datetime import datetime

def run_all_experiments():
    # Récupérer et afficher les informations système
    print("\n" + "="*50)
    print("INFORMATIONS SYSTÈME")
    print("="*50)
    # CPU info
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} (Physical), {psutil.cpu_count()} (Logical)")
    print(f"CPU Utilization: {psutil.cpu_percent(interval=1)}%")
    
    # RAM info
    ram = psutil.virtual_memory()
    print(f"RAM Total: {ram.total / (1024**3):.2f} GB")
    print(f"RAM Available: {ram.available / (1024**3):.2f} GB ({ram.percent}% used)")
    
    # GPU info
    if tf.config.list_physical_devices('GPU'):
        print("\nGPU Information:")
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            print(f"  {gpu.name}")
        print(f"TensorFlow using GPU: Yes")
    else:
        print("\nTensorFlow using GPU: No (CPU only)")
    
    # Afficher la version de TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    
    # Démarrer le chronomètre global
    global_start_time = time.time()
    
    print("\n" + "="*50)
    print("CHARGEMENT DES DONNÉES")
    print("="*50)
    print("Chargement des données MNIST...")
    start_time = time.time()
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data('/mnt/c/Users/Duck/Documents/Cours/M2 MBDS/Industry 4.0/Session3/MNIST/trainingSet/trainingSet/')
    load_time = time.time() - start_time
    print(f"Données chargées en {load_time:.2f} secondes")
    
    # Créer le dataset de test
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
    
    # Analyse des données MNIST
    print("\nAnalyse des données MNIST:")
    print(f"Nombre d'images d'entraînement: {len(X_train)}")
    print(f"Nombre d'images de test: {len(X_test)}")
    print(f"Dimensions des images: {input_shape}")
    
    # Distribution des classes
    y_classes = np.argmax(y_train, axis=1)
    class_counts = np.bincount(y_classes)
    print("\nDistribution des classes dans l'ensemble d'entraînement:")
    for i, count in enumerate(class_counts):
        print(f"  Classe {i}: {count} images ({count/len(y_train)*100:.2f}%)")
    
    # Configurations des expériences
    configurations = [
        # Nombre de rounds fédérés
        {'name': 'Rounds_3', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3, 'rounds': 3},
        {'name': 'Rounds_5', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3, 'rounds': 5},
        {'name': 'Rounds_10', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3, 'rounds': 10},
        
        # Nombre de clients
        {'name': 'Clients_5', 'num_clients': 5, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3, 'rounds': 5},
        {'name': 'Clients_10', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3, 'rounds': 5},
        {'name': 'Clients_20', 'num_clients': 20, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3, 'rounds': 5},
        
        # Distribution des données
        {'name': 'IID_10clients', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3, 'rounds': 5},
        {'name': 'NonIID_10clients', 'num_clients': 10, 'distribution': 'non_iid', 'algo': 'fedavg', 'epochs': 3, 'rounds': 5},
        {'name': 'NonIID_extreme', 'num_clients': 10, 'distribution': 'non_iid_extreme', 'algo': 'fedavg', 'epochs': 3, 'rounds': 5},
        
        # Époques locales
        {'name': 'Epochs_1', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 1, 'rounds': 5},
        {'name': 'Epochs_3', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3, 'rounds': 5},
        {'name': 'Epochs_5', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 5, 'rounds': 5},
        
        # Algorithmes d'agrégation
        {'name': 'FedAvg_10', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3, 'rounds': 5},
        {'name': 'FedSGD_LR0.01', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedsgd', 'epochs': 3, 'rounds': 5, 'lr': 0.01},
        {'name': 'FedSGD_LR0.1', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedsgd', 'epochs': 3, 'rounds': 5, 'lr': 0.1},
        {'name': 'FedProx_mu0.01', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedprox', 'epochs': 3, 'rounds': 5, 'mu': 0.01},
        {'name': 'FedProx_mu0.1', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedprox', 'epochs': 3, 'rounds': 5, 'mu': 0.1},
    ]
    
    # Résultats détaillés pour suivre la progression
    detailed_results = []
    
    # Tableau pour stocker les résultats
    results = []
    
    # Référence: modèle entraîné de manière centralisée
    print("\n" + "="*50)
    print("MODÈLE CENTRALISÉ (RÉFÉRENCE)")
    print("="*50)
    start_time = time.time()
    central_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    central_dataset = central_dataset.shuffle(len(y_train))
    central_dataset = central_dataset.batch(32)
    
    central_model = fl_model.MyModel(input_shape, nbclasses=10)
    
    # Historique pour le modèle centralisé
    centralized_history = {'loss': [], 'accuracy': []}
    for epoch in range(10):
        print(f"Epoch {epoch+1}/10")
        central_model.fit_it(trains=central_dataset, epochs=1, tests=test_dataset, verbose=0)
        loss, acc = central_model.evaluate(test_dataset, verbose=0)
        centralized_history['loss'].append(loss)
        centralized_history['accuracy'].append(acc)
        print(f"  Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    central_loss, central_acc = central_model.evaluate(test_dataset, verbose=0)
    central_time = time.time() - start_time
    
    print(f"\nModèle centralisé - Loss: {central_loss:.4f}, Accuracy: {central_acc:.4f}, Time: {central_time:.2f}s")
    results.append({
        'name': 'Centralized',
        'loss': central_loss,
        'accuracy': central_acc,
        'time': central_time,
        'history': centralized_history
    })
    
    # Exécuter toutes les expériences
    round_metrics = {}
    for config in configurations:
        print("\n" + "="*50)
        print(f"EXPÉRIENCE: {config['name']}")
        print("="*50)
        start_time = time.time()
        
        # Informations sur la configuration
        print(f"Configuration:")
        print(f"  Nombre de clients: {config['num_clients']}")
        print(f"  Distribution des données: {config['distribution']}")
        print(f"  Algorithme: {config['algo']}")
        print(f"  Époques locales: {config['epochs']}")
        
        # Créer la distribution des données
        dist_start = time.time()
        if config['distribution'] == 'iid':
            edges = data_partition.iid_partition(X_train, y_train, config['num_clients'])
        else:
            edges = data_partition.non_iid_partition(X_train, y_train, config['num_clients'])
        dist_time = time.time() - dist_start
        print(f"Distribution des données terminée en {dist_time:.2f}s")
        
        # Analyser la distribution des données
        print("\nAnalyse de la distribution des données par client:")
        for i, edge_name in enumerate(list(edges.keys())[:3]):  # Afficher seulement les 3 premiers pour économiser l'espace
            edge_data = edges[edge_name]
            num_batches = tf.data.experimental.cardinality(edge_data).numpy()
            batch_size = next(iter(edge_data))[0].shape[0]
            num_samples = num_batches * batch_size
            print(f"  Client {edge_name}: ~{num_samples} échantillons")
            
            # Pour les configurations non-IID, montrer la distribution des classes
            if config['distribution'] == 'non_iid' and i < 3:  # Limiter à 3 clients
                # Récupérer les étiquettes
                edge_labels = []
                for batch in edge_data:
                    edge_labels.extend(np.argmax(batch[1].numpy(), axis=1))
                
                edge_class_counts = np.bincount(edge_labels, minlength=10)
                print(f"    Distribution des classes: ", end="")
                for cls, count in enumerate(edge_class_counts):
                    if count > 0:
                        print(f"{cls}:{count} ", end="")
                print()
        
        if len(edges) > 3:
            print(f"  ... et {len(edges) - 3} autres clients")
        
        # Sélectionner l'algorithme d'agrégation
        if config['algo'] == 'fedavg':
            agg_fn = lambda x: aggregation.fedavg(x, 
                central_weights=federated_model.get_weights(), 
                config_name=config['name'], 
                round_num=round_num+1)
        elif config['algo'] == 'fedsgd':
            # Pour FedSGD, nous devons adapter l'interface
            agg_fn = lambda x: aggregation.fedsgd(federated_model, x, 
                central_weights=federated_model.get_weights(),
                config_name=config['name'], 
                round_num=round_num+1)
        elif config['algo'] == 'fedprox':
            # Pour FedProx, nous devons conserver les poids globaux
            global_weights = federated_model.get_weights()
            agg_fn = lambda x: aggregation.fedprox(x, global_weights, 
                config_name=config['name'], 
                round_num=round_num+1)        
        # Créer et configurer le modèle fédéré
        federated_model = fl_model.MyModel(input_shape, nbclasses=10)
        
        # Nombre de rounds fédérés
        num_rounds = 5
        round_history = {'round': [], 'loss': [], 'accuracy': []}
        
        # Exécuter l'apprentissage fédéré
        for round_num in range(num_rounds):
            round_start = time.time()
            print(f"\nRound {round_num+1}/{num_rounds}")
            
            # Exécuter un round d'apprentissage fédéré
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
            
            # Évaluer après chaque round
            round_loss, round_acc = federated_model.evaluate(test_dataset, verbose=0)
            round_time = time.time() - round_start
            
            # Enregistrer les métriques du round
            round_history['round'].append(round_num + 1)
            round_history['loss'].append(round_loss)
            round_history['accuracy'].append(round_acc)
            
            print(f"  Loss: {round_loss:.4f}, Accuracy: {round_acc:.4f}, Time: {round_time:.2f}s")
                    
        # Évaluer le modèle fédéré final
        fed_loss, fed_acc = federated_model.evaluate(test_dataset, verbose=0)
        elapsed_time = time.time() - start_time
        
        print(f"\n{config['name']} - Loss: {fed_loss:.4f}, Accuracy: {fed_acc:.4f}, Time: {elapsed_time:.2f}s")
        
        # Enregistrer les résultats finaux
        results.append({
            'name': config['name'],
            'loss': fed_loss,
            'accuracy': fed_acc,
            'time': elapsed_time,
            'history': round_history
        })
        
        # Enregistrer tous les résultats détaillés
        for i, (loss, acc) in enumerate(zip(round_history['loss'], round_history['accuracy'])):
            detailed_results.append({
                'experiment': config['name'],
                'round': i + 1,
                'loss': loss,
                'accuracy': acc,
                'num_clients': config['num_clients'],
                'distribution': config['distribution'],
                'algo': config['algo'],
                'epochs': config['epochs']
            })
        
        # Libérer la mémoire entre les expériences
        gc.collect()
    
    # Temps total d'exécution
    total_time = time.time() - global_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTemps total d'exécution: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Afficher les résultats sous forme de tableau
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'history'} for r in results])
    print("\n" + "="*50)
    print("RÉSULTATS DE TOUTES LES EXPÉRIENCES")
    print("="*50)
    print(results_df)
    
    # Enregistrer les résultats
    results_df.to_csv('federated_learning_results.csv', index=False)
    
    # Créer un dataframe pour les résultats détaillés
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('federated_learning_detailed_results.csv', index=False)
    
    # Créer le dossier pour les graphiques
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    print("\n" + "="*50)
    print("GÉNÉRATION DES GRAPHIQUES")
    print("="*50)
    
    # 1. Performance globale (précision) par configuration
    plt.figure(figsize=(12, 8))
    sns.barplot(x='name', y='accuracy', data=results_df)
    plt.title('Précision par configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Précision')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/global_accuracy.png')
    plt.close()
    print(f"Graphique enregistré: {plots_dir}/global_accuracy.png")
    
    # 2. Temps d'exécution par configuration
    plt.figure(figsize=(12, 8))
    sns.barplot(x='name', y='time', data=results_df)
    plt.title('Temps d\'exécution par configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Temps (secondes)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/execution_time.png')
    plt.close()
    print(f"Graphique enregistré: {plots_dir}/execution_time.png")
    
    # 3. Progression de la précision par round pour chaque configuration
    plt.figure(figsize=(12, 8))
    for result in results:
        if 'history' in result and 'round' in result['history']:
            plt.plot(result['history']['round'], result['history']['accuracy'], marker='o', label=result['name'])
    plt.title('Progression de la précision par round')
    plt.xlabel('Round')
    plt.ylabel('Précision')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/accuracy_progression.png')
    plt.close()
    print(f"Graphique enregistré: {plots_dir}/accuracy_progression.png")
    
    # 4. Comparaison des temps en fonction du nombre de clients
    clients_data = results_df[results_df['name'].str.contains('Clients_')]
    if not clients_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='name', y='time', data=clients_data)
        plt.title('Temps d\'exécution vs Nombre de clients')
        plt.xlabel('Nombre de clients')
        plt.ylabel('Temps (secondes)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/clients_vs_time.png')
        plt.close()
        print(f"Graphique enregistré: {plots_dir}/clients_vs_time.png")
    
    # 5. Comparaison de précision en fonction du nombre d'époques
    epochs_data = results_df[results_df['name'].str.contains('Epochs_')]
    if not epochs_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='name', y='accuracy', data=epochs_data)
        plt.title('Précision vs Nombre d\'époques locales')
        plt.xlabel('Nombre d\'époques')
        plt.ylabel('Précision')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/epochs_vs_accuracy.png')
        plt.close()
        print(f"Graphique enregistré: {plots_dir}/epochs_vs_accuracy.png")
    
    # 6. Comparaison des algorithmes d'agrégation
    algo_data = results_df[results_df['name'].str.contains('Fed')]
    if not algo_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='name', y='accuracy', data=algo_data)
        plt.title('Précision vs Algorithme d\'agrégation')
        plt.xlabel('Algorithme')
        plt.ylabel('Précision')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/algorithm_vs_accuracy.png')
        plt.close()
        print(f"Graphique enregistré: {plots_dir}/algorithm_vs_accuracy.png")
    
    print("\nTous les résultats et graphiques ont été enregistrés.")

    # 7. Analyse de la convergence en fonction du nombre de rounds
    rounds_data = detailed_df[detailed_df['experiment'].str.contains('Rounds_')]
    if not rounds_data.empty:
        plt.figure(figsize=(12, 8))
        
        for exp_name in rounds_data['experiment'].unique():
            exp_data = rounds_data[rounds_data['experiment'] == exp_name]
            plt.plot(exp_data['round'], exp_data['accuracy'], marker='o', label=exp_name)
        
        plt.title('Convergence en fonction du nombre de rounds')
        plt.xlabel('Round')
        plt.ylabel('Précision')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/convergence_by_rounds.png')
        plt.close()
        print(f"Graphique enregistré: {plots_dir}/convergence_by_rounds.png")

    # 8. Analyse distribution IID vs non-IID
    iid_data = detailed_df[detailed_df['experiment'].str.contains('IID')]
    if not iid_data.empty:
        plt.figure(figsize=(12, 8))
        
        for exp_name in iid_data['experiment'].unique():
            exp_data = iid_data[iid_data['experiment'] == exp_name]
            plt.plot(exp_data['round'], exp_data['accuracy'], marker='o', label=exp_name)
        
        plt.title('Impact de la distribution des données (IID vs non-IID)')
        plt.xlabel('Round')
        plt.ylabel('Précision')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/iid_vs_noniid_impact.png')
        plt.close()
        print(f"Graphique enregistré: {plots_dir}/iid_vs_noniid_impact.png")

    # 9. Analyse de l'effet du nombre d'époques locales
    epochs_data = detailed_df[detailed_df['experiment'].str.contains('Epochs_')]
    if not epochs_data.empty:
        plt.figure(figsize=(12, 8))
        
        for exp_name in epochs_data['experiment'].unique():
            exp_data = epochs_data[epochs_data['experiment'] == exp_name]
            plt.plot(exp_data['round'], exp_data['accuracy'], marker='o', label=exp_name)
        
        plt.title('Effet du nombre d\'époques locales sur la convergence')
        plt.xlabel('Round')
        plt.ylabel('Précision')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/local_epochs_effect.png')
        plt.close()
        print(f"Graphique enregistré: {plots_dir}/local_epochs_effect.png")
    
    from aggregation import generate_progression_visualizations

    # Générer les visualisations de progression
    print("\n" + "="*50)
    print("GÉNÉRATION DES VISUALISATIONS DE PROGRESSION")
    print("="*50)
    generate_progression_visualizations()

if __name__ == "__main__":
    run_all_experiments()