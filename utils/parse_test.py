import sqlparse

sql="SELECT unique_carrier, COUNT(*) FROM flights WHERE origin_state_abr='LA' GROUP BY unique_carrier"
stmt_parsed=sqlparse.parse(sql)
for token in stmt_parsed[0].tokens:
    print(token)
    print(str(token))