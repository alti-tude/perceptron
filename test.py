from joblib import Memory

mem = Memory(cachedir='./tmp', verbose=1)

@mem.cache
def sq(x):
    return x*x

sq(2)
sq(100)
sq(101)