#  Llamado de las librerias requeridas
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

class Regresion:    

    def __init__(self):

        #lectura de los datos 
        self.df=pd.read_excel(os.getcwd()+'\MAESTRIA\MODELOS_PREDICTVOS\data_weekly_close_nya.xlsx')

    def modelo_regresion(self):
        # Data: Period (X) anyd NYA valores (y)
        period=self.df[['PERIOD']]
        nya_values=self.df[['NYA']]

        # Crea modelo Regresion lineal
        model = LinearRegression()


        # Ajusta el modelo
        model.fit(period, nya_values)
        x=model.fit(period, nya_values)


        #Pronóstico
        predictions = model.predict(period)

        #Indicadores del modelo 
        mse = mean_squared_error(nya_values, predictions)
        r2 = r2_score(nya_values, predictions)
        return period,nya_values,predictions,mse,r2,x


    def grafica(self):
       
         # Grafica
        plt.figure(figsize=(10, 6))
        plt.plot(period, nya_values, color='blue', label='Actual NYA values')
        plt.plot(period, predictions, color='red', linewidth=1, label='Pronostico NYA', linestyle='--')
        plt.xlabel('Periodo')
        plt.ylabel('NYA valores')
        plt.title('Regresion Lineal: NYA Index Actual vs Pronostico')
        plt.grid(visible=True)
        plt.legend()
        plt.show()


#  *** Main ***
if __name__ == '__main__':
        obj=Regresion()
        period,nya_values,predictions,mse,r2,x=obj.modelo_regresion()
        X_with_const = sm.add_constant(period)  # Adds a constant term to the predictor
        ols_model = sm.OLS(nya_values, X_with_const).fit()
        print(ols_model.summary())
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"R^2 Score: {r2:.6f}")
        print("Intercept:" , x.intercept_ )
        print("Coefficients: ",x.coef_ )
        print("gráfica del modelo: \n",obj.grafica())



       