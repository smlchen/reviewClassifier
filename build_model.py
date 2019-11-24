from keras.models import Sequential
from keras.layers import Dense


def build_model(num_hidden_layers, num_nodes, input_dim, output_dim, activation= 'relu', output_activation = 'softmax', optimizer = 'adam', initializer = 'glorot_uniform', loss = 'categorical_crossentropy', metrics=['accuracy']):
    
    #create the NN
    #with num_hidden_layers each with num_nodes
    model = Sequential()
    #first hidden layer
    model.add(Dense(num_nodes, input_dim=input_dim, activation=activation, kernel_initializer=initializer, bias_initializer=initializer))
    for i in range(num_hidden_layers-1): # additional hidden layers
        model.add(Dense(num_nodes, activation=activation, kernel_initializer=initializer, bias_initializer=initializer)) 
    
    #output layer has output_dim nodes
    model.add(Dense(output_dim, activation=activation, kernel_initializer=initializer, bias_initializer=initializer)) 
    model.compile(optimizer= optimizer, loss = loss, metrics=metrics)
    return model
    
