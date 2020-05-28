def initcap(m):
    return m[:1].upper()+m[1:].lower()


def instr(a,b):
    ls=[]
    for i in range(len(a)):
        ls.append(a[i])
        if ls[i]==b:
            return i+1
    return 0

def connect(data) :
    import cx_Oracle
    import pandas as pd

    dsn = cx_Oracle.makedsn('localhost',1521,'orcl') # 오라클에 대한 주소정보
    db = cx_Oracle.connect('scott','tiger',dsn)      # 오라클 접속 유저 정보 (이름, 비밀번호, 주소)
    cursor = db.cursor()                             # 데이터를 담을 메모리 이름을 cursor로 선언
    cursor.execute("""select * from %s""" %data)          # """SQL Query작성""" 한 결과가 cursor 메모리에 담김
    row = cursor.fetchall()                          # cursor 메모리에 담긴 데이터를 한 행씩 가져옴
    colname = cursor.description                     # 위에서 SELECT한 테이블의 컬럼명을 가져옴
    cursor.close()                                   # cursor 메모리를 닫음
    col = []                                         # list 생성
    for i in colname :
        col.append(i[0])                             # 테이블 컬럼명을 채워넣음
    select_data = pd.DataFrame(row)                          # 데이터를 테이블화(Data Frame)함
    select_data = pd.DataFrame(row,columns=col)

    return select_data