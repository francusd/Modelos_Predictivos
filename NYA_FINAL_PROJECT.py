# Llamado de las librerias requeridas
import os 
import numpy as np
import pandas as pd 
import sweetviz as sv
from ydata_profiling import ProfileReport



#Definición de la clase
class DataProcesador: 
    
 #constructor   
 def __init__(self):   
    #Obtencion de rutas de los archivos CSV
   self.datos=os.getcwd()+'\MAESTRIA\MODELOS_PREDICTVOS\WEEKLY_NYA.csv'


#funcion para hacer el join de ambos dataset
 def lectura_dataset(self):
    
    #lectura de dataset
    df=pd.read_csv(self.datos)
    
    #transformacion de datos
    df['Date'] = pd.to_datetime(df['Date'])
    
    #transformaciones 
    df['Period'] = df['Date'].dt.strftime('%Y-%m')
    df['Años'] = df['Date'].dt.strftime('%Y')
    df=df.where(df['Años']>'2022')  
    df=df.dropna(how='all')
    df['Index']='NYA'
    df['Exchange']='New York Exchange'
    df['Country']='United States'
    df['Currency']='USD dollars'

    return df
 


#Genera EDA automatizado
 def analisis_exploratorio(self):
   df=pd.read_excel(os.getcwd()+'\MAESTRIA\MODELOS_PREDICTVOS\data_weekly_close_nya.xlsx')
                  
   # Genera el eda Sweetviz
   #report = sv.analyze(df)
   # guardar el reporte
   #report.show_html('nya_sweetviz_report.html')
   profile = ProfileReport(df, title="Analisis Exploratorio de Datos: Index Stocks")
   profile.to_file(os.getcwd()+'\MAESTRIA\MODELOS_PREDICTVOS\EDA_INDEX_NYA.html')


 def exportacion_file(self):
   final_df,descripcion = self.lectura_dataset()

   with pd.ExcelWriter(os.getcwd()+'\MAESTRIA\MODELOS_PREDICTVOS\INDEX_NYA_WEEKLY_final.xlsx') as writer:
      final_df.to_excel(writer,sheet_name='DATOS', index=False)
      descripcion.to_excel(writer,sheet_name='DESCRIPCION')
      

#  *** Main ***
if __name__ == '__main__':
      obj=DataProcesador()
      
      dataset=obj.lectura_dataset()
      #muestreo del dataset 
      print("\n\n Muestreo del dataset \n\n\n",dataset.head())
      
      #Informacion del dataset
      print("Dataset informacion:\n",dataset.info())

      #Descripcion de cada columna del dataset
      print("\n\nDataset Descripcion:\n",dataset.describe(include='all'))

      obj.exportacion_file()
      obj.analisis_exploratorio()
