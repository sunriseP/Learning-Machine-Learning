import pandas as pd
import numpy as np

df = pd.read_csv('patients.csv')

ages = df['Age']
heights = df['Height_cm']
weights = df['Weight_kg']

d1 = ages - np.mean(ages)
d2 = heights - np.mean(heights)
d3 = weights - np.mean(weights)

# compute the Pearson correlation coefficients
fz = np.sum(d1 * d2)
fm = np.sqrt( np.sum(d1**2) * np.sum(d2**2) )
r_ah = fz / fm

fz = np.sum(d1 * d3)
fm = np.sqrt( np.sum(d1**2) * np.sum(d3**2) )
r_aw = fz / fm

print(f'Age and Height: {r_ah}')
print(f'Age and Weight: {r_aw}')