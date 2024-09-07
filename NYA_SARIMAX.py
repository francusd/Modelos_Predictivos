import os
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error,r2_score
import matplotlib.pyplot as plt



#lectura de los datos 
data=pd.read_excel(os.getcwd()+'\MAESTRIA\MODELOS_PREDICTVOS\data_weekly_close_nya.xlsx')

# Set 'PERIOD' as the index
data.set_index('PERIOD', inplace=True)

# Define the order of the SARIMA model (p,d,q)(P,D,Q,s)
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

# Fit SARIMA model
model = SARIMAX(data['NYA'], order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit()

# Summary of the model
print(results.summary())



#Hacer las predicciones
y_pred = results.get_forecast(len(data.index))
y_pred_df = y_pred.conf_int(alpha = 0.05)
y_pred_df["Predictions"] = results.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = data.index
y_pred_out = y_pred_df["Predictions"]



#Calculo de Metricas de Evaluacion del Modelo
mae_sar = mean_absolute_error(data['NYA'],y_pred_out)
mse_sar = mean_squared_error(data['NYA'],y_pred_out)
mad_sar = data['NYA'].sub(y_pred_out).abs().mean()
mape_sar = (data['NYA'].sub(y_pred_out).abs() / data['NYA']).mean() * 100

ts_sar = mae_sar / mad_sar
#ts_min_sar = min(ts_w, 0)
#ts_max_sar = max(ts_w, 0)

r2_sar = r2_score(data['NYA'], y_pred_out)

print(f'R^2 Score: {r2_sar:.2f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape_sar:.2f}%')
print(f'Mean Absolute Deviation (MAD): {mad_sar:.2f}')
print(f'Mean Absolute Error = {mae_sar:.2f}')
print(f'Mean Squared Error = {mse_sar:.2f}')
#print(f'TSmin: {ts_min_sar}')
#print(f'TSmax: {ts_max_sar}')


#Graficar las Pronostico junto a los valores reales
plt.plot(data['NYA'], color = 'blue', label = 'NYA Index')
plt.plot(y_pred_out, color='red', label = 'SARIMAX Pronostico', linestyle='--',linewidth=1 )
plt.ylabel('NYA values')
plt.xlabel('PERIOD')
plt.xticks(rotation=45)
plt.title("SARIMAX Model: NYA Index vs Pronostico")
plt.rcParams["figure.figsize"] = (16,5)
plt.grid(visible=True)
plt.legend()
plt.show()