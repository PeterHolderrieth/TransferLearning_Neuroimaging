import json
with open("test.json", "r") as read_file:
    data = json.load(read_file)


print(data)
print(type(data))
print(data['some number'])
print(type(data['some number']))

print(data['age'])
print(type(data['age']))

dictionary={
'age': 58,
'number': 1.45,
'greeting': 'hello',
'lasso': [1.3,5.6,4.]
}
with open("example.json", "w") as write_file:
    json.dump(dictionary, write_file,indent=2)