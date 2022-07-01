import torch
import torch.nn as nn
torch.manual_seed(0)

class EntityCat(nn.Module):
    def __init__(self, embedding_size, num_numerical_cols,
                 output_size, neurons_in_layers=[100], p=0.4):
        '''
        embedding_size: Contains the embedding size for the categorical columns
        num_numerical_cols: Stores the total number of numerical columns
        output_size: The size of the output layer or the number of possible outputs.
        neurons_in_layers: List which contains number of neurons for all the neurons_in_layers.
        p: Dropout with the default value of 0.5

        '''
        super(EntityCat, self).__init__()
        self.embedding_size_len = len(embedding_size)
        # list of ModuleList objects for all categorical columns
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])

        # drop out value for all neurons_in_layers
        self.embedding_dropout = nn.Dropout(p)

        # list of 1 dimension batch normalization objects for all numerical columns
        #         self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        # the number of categorical and numerical columns are added together and stored in input_size
        all_layers = nn.ModuleList()
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        # loop iterates to add corresonding neurons_in_layers to all_layers list above
        for num_neurons in neurons_in_layers:
            all_layers.append(nn.Linear(input_size, num_neurons))
            all_layers.append(nn.ReLU(inplace=True))
            #             all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = num_neurons

        # append output layer to list of neurons_in_layers
        all_layers.append(nn.Linear(neurons_in_layers[-1], output_size))

        # pass all neurons_in_layers to the sequential class
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical):
        # this starts the embedding of categorical columns
        embeddings = []
        assert self.embedding_size_len == x_categorical.shape[1], \
        'The number of features in the feature list and x_categorical shape should match'
        for i, e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)
        x = torch.cat([x], 1)
        x = self.layers(x)
        x = torch.sigmoid(x)
        return x


class EntityCat_num(nn.Module):
    def __init__(self, embedding_size, num_numerical_cols,
                 output_size, neurons_in_layers=[100], p=0.4):
        '''
        embedding_size: Contains the embedding size for the categorical columns
        num_numerical_cols: Stores the total number of numerical columns
        output_size: The size of the output layer or the number of possible outputs.
        neurons_in_layers: List which contains number of neurons for all the neurons_in_layers.
        p: Dropout with the default value of 0.5

        '''
        super(EntityCat_num, self).__init__()
        self.embedding_size_len = len(embedding_size)
        # list of ModuleList objects for all categorical columns
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf)\
                 for ni, nf in embedding_size])

        # drop out value for all neurons_in_layers
        self.embedding_dropout = nn.Dropout(p)

        # list of 1 dimension batch normalization objects for all numerical columns
        #         self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        # the number of categorical and numerical columns are added together and stored in input_size
        all_layers = nn.ModuleList()
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols
        # print('hao-----', len(input_size))
        # loop iterates to add corresonding neurons_in_layers to all_layers list above
        for num_neurons in neurons_in_layers:
            all_layers.append(nn.Linear(input_size, num_neurons))
            all_layers.append(nn.ReLU(inplace=True))
            #             all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = num_neurons

        # append output layer to list of neurons_in_layers
        all_layers.append(nn.Linear(neurons_in_layers[-1], output_size))

        # pass all neurons_in_layers to the sequential class
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical):
        # this starts the embedding of categorical columns
        embeddings = []
        # assert self.embedding_size_len == x_categorical.shape[1], \
        # 'The number of features in the feature list and x_categorical shape should match'
        for i, e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:, i].long()))
        embeddings.append(x_categorical[:,i+1:].long())
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)
        x = torch.cat([x], 1)
        x = self.layers(x)
        x = torch.sigmoid(x)
        return x


