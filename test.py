def input_bool(string):
    x=input(string)
    if x=="True":
        return True
    elif x=="False":
        return False
    else: 
        print("Possible values are 'True' or 'False'.")
        return input_bool(string)

def input_float(string):
    x=input(string)
    try:
        x=float(x)
        return(x)
    except ValueError:
        print("Not a float")
        return(input_float(string))

def input_string(string):
    x=input(string)
    try:
        x=str(x)
        return(x)
    except ValueError:
        print("Not a string")
        return(input_float(string))

def input_int(string):
    x=input(string)
    try:     
        x=float(x)
        if x.is_integer():
            return(int(x))
        else: 
            print("An int, not a float, is required.")
            return(input_int(string))
    except ValueError:
        print("Not an int.")
        return(input_int(string))

def my_input(string,type_):
    if type_=='int':
        return input_int(string)
    elif type_=='float':
        return input_float(string)
    elif type_=='bool':
        return input_bool(string)
    elif type_=='string':
        return input_string(string)
    elif type_=='int_list':
        return input_int_list(string)
    elif type_=='float_list':
        return input_float_list(string)
    else: 
        sys.exit("Unknown input type.")

y=input_string("Hello: ")
print(type(y))
print(y)