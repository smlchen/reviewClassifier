import numpy as np
import pandas as pd
import time
from build_model import build_model
from save_grid_search_results import save_grid_search_results

#saves all models, history and results in ./grid_search_results

def grid_search(X_train, y_train, X_test, y_test, num_hidden_layers_list=[1], num_nodes_list=[4], epochs_list=[50], batch_size_list=[32], verbose=2, \
    activation= 'relu', output_activation = 'softmax', optimizer = 'adam', initializer = 'glorot_uniform',\
        loss = 'categorical_crossentropy',  metrics=['accuracy'], callbacks=None, class_weight=None, save_model=False, predict_rating=True):
    
    input_dim = np.shape(X_train)[1]
    output_dim = len(np.unique(np.argmax(y_train, axis=1)))
    weighted = 'False' if not class_weight else 'True'
    predict = 'rating' if predict_rating else 'category'
    path = r'./grid_search_results/' 
    
    #grid of all combinations of layers, nodes, epochs, batch_size 
    for num_hidden_layers in num_hidden_layers_list:
        for num_nodes in num_nodes_list:
            for epochs in epochs_list:
                for batch_size in batch_size_list:
                    print('layers:', num_hidden_layers, '\nnodes:', num_nodes)
                    #build the model
                    model = build_model(num_hidden_layers, num_nodes, input_dim, output_dim, activation=activation, output_activation=output_activation, optimizer=optimizer, initializer=initializer, loss=loss, metrics=metrics)
                    #fit model
                    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks, validation_data=(X_test, y_test), class_weight=class_weight)
                    #evaluate model on testing DataFrame
                    eval_loss, accuracy = model.evaluate(X_test, y_test, verbose=verbose)
                    #save model and history
                    file_name = ''
                    if save_model:
                        file_name = str(int(time.time()))
                        model.save(path + file_name + '.h5')
                        pd.DataFrame(history.history).to_csv(path + file_name + '.history', index=False)
                    
                    #save results
                    results = save_grid_search_results(dict(file_name=file_name+'.h5', predict=predict, predictor_type='neural_net', layers=num_hidden_layers,nodes=num_nodes, activation= str(activation) + r'/' + str(output_activation),\
                        optimizer=optimizer,loss_function=loss, epochs=epochs, batch_size=batch_size, weighted=weighted, loss=eval_loss, accuracy=accuracy))
    
    return results
