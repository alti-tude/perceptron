# perceptron

## choosing layer config
have 1 hidden layer as the mean of the ip and op -> train 92.7, test 91.2


## to do
1. make notes for nn eqns


## installing the kernal
```
$ python -m venv projectname
$ source projectname/bin/activate
(venv) $ pip install ipykernel
(venv) $ ipython kernel install --user --name=projectname
```

## joblib caching
refer to test.py
```python
from joblib import Memory

mem = Memory(cachedir='./tmp', verbose=1)

@mem.cache
def sq(x):
    return x*x

sq(2)
sq(100)
sq(101)
```