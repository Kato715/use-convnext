# convnext

archeological classification project

## creating conda environment


```console
$ conda env create --file environment.yml
$ conda activate archeological
```

## model fine tuning

create label_encoder and csv file
```console
$ python create_csv.py conf/config.json
```

training
```console
$ python train.py conf/config.json
```

test
```console
$ python test.py conf/config.json
```


