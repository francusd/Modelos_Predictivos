#llamado de librerias requeridas
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#lectura de los datos 
df=pd.read_excel(os.getcwd()+'\MAESTRIA\MODELOS_PREDICTVOS\data_weekly_close_nya.xlsx')
period=df['PERIOD']
nya_values=df['NYA']

# Calculate the rolling mean(promedio moviles)
df['Rolling_Mean'] = df['NYA'].rolling(window=4).mean()
predictions=df['Rolling_Mean']
# muestra los datos con el rolling mean(promedio moviles)
print(df)

# Grafica
plt.figure(figsize=(10, 6))
#plt.scatter(period, nya_values, color='blue', label='Actual NYA values')
plt.plot(period, nya_values, color='blue', label='Actual NYA values')
plt.plot(period, predictions, color='red', linewidth=1, label='Predicted NYA values', linestyle='--')
plt.xlabel('Period')
plt.ylabel('NYA values')
plt.title('Rolling Mean Forecasting: Actual vs Predicted NYA values')
plt.grid(visible=True)
plt.legend()
plt.show()