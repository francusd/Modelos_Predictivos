import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller

#lectura de los datos 
df=pd.read_excel(os.getcwd()+'\MAESTRIA\MODELOS_PREDICTVOS\data_weekly_close_nya.xlsx')

# Step 3: Perform the ADF test
result = adfuller(df['NYA'])

# Step 4: Print the results
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'   {key}: {value}')