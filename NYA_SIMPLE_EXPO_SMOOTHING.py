
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing



#lectura de los datos 
df=pd.read_excel(os.getcwd()+'\MAESTRIA\MODELOS_PREDICTVOS\data_weekly_close_nya.xlsx')


model = SimpleExpSmoothing(df['NYA'])
fit = model.fit(smoothing_level=1, optimized=False)

# Generate the forecast
df['Forecast'] = fit.fittedvalues


plt.figure(figsize=(10,6))
plt.plot(df['PERIOD'], df['NYA'],color='blue', label='NYA Index valor')
plt.plot(df['PERIOD'], df['Forecast'],color='red', label='Pronostico', linestyle='--',linewidth=1)
plt.xlabel('PERIODO')
plt.ylabel('NYA valores')
plt.title('Suavizamiento Exponencial Simple:\nPronostico NYA INDEX')
plt.grid(visible=True)
plt.legend()
plt.show()

# Print the fitted values and forecast
print(df)
print(fit.summary())
