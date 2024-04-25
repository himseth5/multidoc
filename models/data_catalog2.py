from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import psycopg2
import pandas as pd
import yaml
from llm.qna import QnA, ExlAI
from models.ColumnDescription import ColumnDescription
from models.program_builder import program
import time



class db_conn:
      
      def __init__(self,conn)->None:
            self.conn=conn
        

      def getDatabasesInSystem(self)->list:
            # try:
                
                query4 = '''SELECT datname FROM pg_database
            WHERE datistemplate = false; '''
                cursor = self.conn.cursor()
                cursor.execute(query4)
                data10 = cursor.fetchall()
                dbNameList = []
                for i in data10:
                    dbNameList.append(i[0])
                return dbNameList 
      
      def getTablesFromdb(self):
        # try:
        query1 = '''SELECT table_name FROM information_schema.tables
        WHERE table_schema='public'
        ORDER BY table_schema,table_name;'''
        cursor = self.conn.cursor()    
        cursor.execute(query1)
        data1=cursor.fetchall()
        # print(data1)
        tablenameslist = []
        for i in data1:
            tablenameslist.append(i[0])
        return tablenameslist
      
      def getcolumndata(self,tablenameslist)->dict:
           tablecolumnsdict = {}
           for table_name in tablenameslist:
                query = f'''select column_name, data_type
                from INFORMATION_SCHEMA.COLUMNS 
                where table_name = '{table_name}';'''
                cursor = self.conn.cursor()    
                cursor.execute(query)
                data2=cursor.fetchall()
                tablecolumnsdict[table_name] = data2
                return tablecolumnsdict
           
      def getcoldescription(self,tablecolumnsdict)->pd.DataFrame:
           with open('prompt.txt','r') as f:
                prompt = f.read()

           llm = ExlAI()
           describingProgram = program(ColumnDescription,llm)
           df = pd.DataFrame(columns=['TableName', 'ColumnName', 'DataType','ColumnDescription'])

           for tableNameKey in tablecolumnsdict:
                tableName = tableNameKey
                for item in tablecolumnsdict[tableNameKey]:
                    columnName = item[0]
                    dataType = item[1]
                    query3 = f'select {columnName} from {tableName} limit 5;'
                    cursor = self.conn.cursor()    
                    cursor.execute(query3)
                    tableRow=cursor.fetchall()
                    dataRow = []
                    for item in tableRow:
                        dataRow.append(item[0])
                    respond = describingProgram.programrespond(prompt)
                    response = respond(table_name=tableName,column_name=columnName,data_type=dataType,data_rows=dataRow)
                    # print(response.description)
                    new_row = {'TableName': tableName, 'ColumnName': columnName,'DataType':dataType,'ColumnDescription':response.description}
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    df = df.reset_index(drop=True)
                    # break
                print(f"Evaluation complete for  {tableName}")
                time.sleep(5)
           return df




    
