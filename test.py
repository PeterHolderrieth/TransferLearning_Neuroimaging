def testing(x:int):
    return x**2

def execute(func: callable, y):
    print(func(y))

execute(testing,2)
