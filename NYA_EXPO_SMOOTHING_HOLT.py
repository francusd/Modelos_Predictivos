import os
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt


#lectura de los datos 
df=pd.read_excel(os.getcwd()+'\MAESTRIA\MODELOS_PREDICTVOS\data_weekly_close_nya.xlsx')
df.set_index('PERIOD', inplace=True)
print(df)


# Fit the model
model = ExponentialSmoothing(df['NYA'], trend='add', seasonal=None)
fit = model.fit()

# Resumen del modelo
print(fit.summary())

# Make predictions
forecast = fit.forecast(steps=4)  # Forecasting next 5 periods


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['NYA'], label='NYA Index',color='blue')
plt.plot(fit.fittedvalues.index, fit.fittedvalues, label='Valor Ajustado', linestyle='--',linewidth=1,color='red')
plt.plot(forecast.index, forecast, label='Pronostico',linewidth=2)
plt.title('Suavizamiento Exponencial Holt:\n NYA Index Pronostico')
plt.xlabel('Periodo')
plt.ylabel('NYA Valores')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()



# Print the forecast
print("\nForecast for the next 5 periods:")
print(forecast)

# Print model parameters
print("\nModel Parameters:")
print(f"Alpha (level): {fit.params['smoothing_level']:.4f}")
print(f"Beta (trend): {fit.params['smoothing_trend']:.4f}")
print(f"Initial Level: {fit.params['initial_level']:.4f}")
print(f"Initial Trend: {fit.params['initial_trend']:.4f}")
