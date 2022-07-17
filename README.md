# NCF

### A pytorch implementation job recommendation system 

## Setup
``
pip install -e . 
``

## Model training 
The training pipe jupyter notebook:

```
./jupyter/train_pipe.ipynb
```

The model script:

```
./src/model_entity.py
```

There are two model classes in the script:  EntityCat, EntityCat_sbert.

The EntityCat is the model having categorical feature input. The EntityCat_sbert using the sbert embedding as a freezed embedding layer.
