#pydantic is a data validationa and data parsing library for python. It uses python type annotations to validate and parse data
# eg in this file we will change the type of name from string to somethign then check

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    
    name : str = 'Aayush' # the default value 
    age : Optional[int] = None # it can be int or None
    email : EmailStr
    cgpa : float = Field(gt=0, lt=10)


# new_student = {'name': 32} # now u can make the validation as name was of string type but we gave it int so it will give error and will not create the object of student class


# new_student = {} #default value 
# new_student = {'name': 'Aayush', 'age': 25} 
new_student = {'name': 'Aayush', 'age': '25'} #type coerceing it will ocnverts the '25' to 25
new_student = {'name': 'Aayush', 'age': 25, 'email': 'abc@gmail.com', 'cgpa' : 7}


student = Student(**new_student) #object of Student class 

print(type(student)) #it makes the pydantic object 

print(student) 