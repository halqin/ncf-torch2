import torch
import torch.nn as nn


class EntityCat(nn.Module):
    def __init__(self, embedding_size, num_numerical_cols,
                 output_size, layers=[100], p=0.4):
        '''
        embedding_size: Contains the embedding size for the categorical columns
        num_numerical_cols: Stores the total number of numerical columns
        output_size: The size of the output layer or the number of possible outputs.
        layers: List which contains number of neurons for all the layers.
        p: Dropout with the default value of 0.5

        '''
        super(EntityCat, self).__init__()
        # list of ModuleList objects for all categorical columns
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])

        # drop out value for all layers
        self.embedding_dropout = nn.Dropout(p)

        # list of 1 dimension batch normalization objects for all numerical columns
        #         self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        # the number of categorical and numerical columns are added together and stored in input_size
        all_layers = nn.ModuleList()
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        # loop iterates to add corresonding layers to all_layers list above
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            #             all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        # append output layer to list of layers
        all_layers.append(nn.Linear(layers[-1], output_size))

        # pass all layers to the sequential class
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical):
        # this starts the embedding of categorical columns
        embeddings = []
        for i, e in enumerate(self.all_embeddings):
            # print('hao-----', x_categorical.shape)
            embeddings.append(e(x_categorical[:, i]))

        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)
        x = torch.cat([x], 1)
        x = self.layers(x)
        x = torch.sigmoid(x)
        return x





