print(1+2)

# 문제1. 기본 문장을 실행하세요!
a=1
b=2
print(a+b)

#%% 영역 구분자
# 문제. for문을 구현하시오!
for i in [1,2,3]:
    print(i)

#%% 문제2. 
    for i in [2,3,4]:
        print(i)

#%% 문제3.
import keyword
print(keyword.kwlist)

#%% 문제4.
and =1;
    
#%%
import csv
import copy

file=open("K:\\Itwill\\2. Python\\자료\\emp2.csv",'r')
emp=csv.reader(file)
file=open("K:\\Itwill\\2. Python\\자료\\dept2.csv","r")
dept=csv.reader(file)

for dept_list in dept:
    for emp_list in emp:
        if dept_list[0]==emp_list[7]:
            print(emp_list[1], dept_list[2])
