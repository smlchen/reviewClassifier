import pandas as pd
import os.path

def save_grid_search_results(path, search_results):
    
    #create or open results df
    if os.path.exists(path + 'results.csv'):
        results = pd.read_csv(path + 'results.csv', header=0)
    else: 
        results = pd.DataFrame(columns=['file_name', 'predict','predictor_type', 'layers', 'nodes', 'activation', 'optimizer', 'loss_function', 'epochs', 'batch_size', 'weighted',  'loss', 'accuracy'])
    
    #add results and save
    results = results.append(search_results, ignore_index=True)
    results.to_csv(path + 'results.csv', index=False)
    
    return results
