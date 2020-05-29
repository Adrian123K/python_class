def mysql(t):
    import pymysql,pandas as pd
    conn = pymysql.connect(host="localhost", user="root",password="1234",
                           db="orcl",charset="utf8")
    curs = conn.cursor()

    sql = f"select * from {t}"
    curs.execute(sql)

    rows = curs.fetchall()
    colname = curs.description

    col = []
    for i in colname:
        col.append(i[0].lower())
    rs = pd.DataFrame(list(rows),columns=col)
    return rs