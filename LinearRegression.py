import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from matplotlib import pyplot as plt


diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

X = df[["age"]]
y = df[["target"]]

linear = LinearRegression().fit(X, y)
b0 = linear.intercept_[0]
b1 = linear.coef_[0][0]

novosX = np.random.uniform(X.min(), X.max(), size=30)
novosX_df = pd.DataFrame(novosX, columns=["age"])
yPrevistos = linear.predict(novosX_df)


plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', label="Dados originais")
plt.plot(X, b0 + b1*X, color='red', label="Reta ajustada")
plt.scatter(novosX_df, yPrevistos, color='cyan', s = 100, alpha = 0.5,  label="Pontos aleatórios previstos")
plt.xlabel("Idade (padronizada)")
plt.ylabel("Progressão da diabetes")
plt.title("Regressão Linear - Diabetes")
plt.legend()
plt.grid(True)
plt.savefig("analise.png", dpi=300)
plt.show()















