from numpy import std
import torch
import torch.nn as nn
import numpy as np
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
        # self.embedding_dropout = nn.Dropout(p)

        # list of 1 dimension batch normalization objects for all numerical columns
        #         self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        # the number of categorical and numerical columns are added together and stored in input_size
        mlp_modules = nn.ModuleList()
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        # loop iterates to add corresonding neurons_in_layers to all_layers list above
        for num_neurons in neurons_in_layers:
            mlp_modules.append(nn.Linear(input_size, num_neurons))
            mlp_modules.append(nn.ReLU(inplace=True))
            #             all_layers.append(nn.BatchNorm1d(i))
            mlp_modules.append(nn.Dropout(p))
            input_size = num_neurons

        # pass all neurons_in_layers to the sequential class
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.predict_layer = nn.Linear(neurons_in_layers[-1], output_size)
        self._init_weight_()

    def _init_weight_(self):
        # nn.init.normal_(self.all_embeddings, std=0.01)
        for m in self.all_embeddings:
            nn.init.normal_(m.weight, std=0.01)
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_normal_(self.predict_layer.weight, a=1)
        

    def forward(self, x_categorical):
        # this starts the embedding of categorical columns
        to_cat = []
        assert self.embedding_size_len == x_categorical.shape[1], \
        'The number of features in the feature list and x_categorical shape should match'
        for col_index, emb_layer in enumerate(self.all_embeddings):
            # print('hao----', e.weight.grad)
            to_cat.append(emb_layer(x_categorical[:, col_index]))
        x = torch.cat(to_cat, 1)
        x = self.mlp_layers(x)
        x = self.predict_layer(x)
        x = torch.sigmoid(x)
        return x


class EntityCat_sbert(nn.Module):
    def __init__(self, embedding_size, num_numerical_cols,
                 output_size, word_weight, encode_array, neurons_in_layers=[100], p=0.4):
        '''
        embedding_size: Contains the embedding size for the categorical columns
        num_numerical_cols: Stores the total number of numerical columns
        output_size: The size of the output layer or the number of possible outputs.
        neurons_in_layers: List which contains number of neurons for all the neurons_in_layers.
        p: Dropout with the default value of 0.4
        '''
        super(EntityCat_sbert, self).__init__()
        self.embedding_size_len = len(embedding_size)
        # list of ModuleList objects for all categorical columns
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.sbert_embeddings = nn.Embedding.from_pretrained(word_weight, freeze=True)
        self.encode_array = encode_array
        self.sorter = np.argsort(self.encode_array)
        # drop out value for all neurons_in_layers
        # self.embedding_dropout = nn.Dropout(p)

        # list of 1 dimension batch normalization objects for all numerical columns
        #         self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        # the number of categorical and numerical columns are added together and stored in input_size
        mlp_modules = nn.ModuleList()
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + word_weight.shape[1]

        # loop iterates to add corresonding neurons_in_layers to all_layers list above
        for num_neurons in neurons_in_layers:
            mlp_modules.append(nn.Linear(input_size, num_neurons))
            mlp_modules.append(nn.ReLU(inplace=True))
            #             all_layers.append(nn.BatchNorm1d(i))
            mlp_modules.append(nn.Dropout(p))
            input_size = num_neurons

        # pass all neurons_in_layers to the sequential class
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.predict_layer = nn.Linear(neurons_in_layers[-1], output_size)
        self._init_weight_()

    def _init_weight_(self):
        # nn.init.normal_(self.all_embeddings, std=0.01)
        for m in self.all_embeddings:
            nn.init.normal_(m.weight, std=0.01)
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_normal_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
        

    def forward(self, x_categorical):
        # this starts the embedding of categorical columns
        to_cat = []
        assert self.embedding_size_len == x_categorical.shape[1], \
        'The number of features in the feature list and x_categorical shape should match'
        for col_index, emb_layer in enumerate(self.all_embeddings):
            # print('hao----', emb_layer.weight.grad)
            to_cat.append(emb_layer(x_categorical[:, col_index]))
        item_index_tensor = self._item_index(x_categorical)
        to_cat.append(self.sbert_embeddings(item_index_tensor))
        x = torch.cat(to_cat, 1)
        x = self.mlp_layers(x)
        prediction = self.predict_layer(x)
        return prediction

    def _item_index(self, x_categorical):
        '''
        The the index from the item enecoder array
        '''
        itemids = x_categorical[:,1].cpu().numpy()
        item_index = self.sorter[np.searchsorted(self.encode_array, itemids, sorter=self.sorter)]
        item_index_tensor = torch.LongTensor(item_index)
        return item_index_tensor


