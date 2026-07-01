from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person= {'name': 'Aayush', 'age': '25'}
new_person2: Person= {'name': 'Aayush', 'age': 25}

print(new_person)
print(new_person2)

#both will works it only defines the structure on hoevr u can see 
    