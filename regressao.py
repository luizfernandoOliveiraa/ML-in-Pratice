import pandas as pd
from sklearn import linear_model
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_excel("./data/dados_cerveja_nota.xlsx", engine="openpyxl")
df.head()

X = df[["cerveja"]]
y = df["nota"]

reg = linear_model.LinearRegression(fit_intercept=True)
tree = tree.DecisionTreeRegressor(random_state=42)

reg.fit(X, y)

a, b = reg.intercept_, reg.coef_[0]

print(f"Equação da reta: y = {a:.2f} + {b:.2f}x")

predict_reg = reg.predict(X.drop_duplicates())

plt.plot(X, y, "o", label="Dados Reais")
plt.plot(X.drop_duplicates(), predict_reg, "r-", label="Reta de Regressão")
plt.grid()
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.legend()
plt.show()

tree.fit(X, y)

predic_tree = tree.predict(X.drop_duplicates())

plt.plot(X, y, "o", label="Dados Reais")
plt.plot(X.drop_duplicates(), predic_tree, "g-", label="Previsão da Árvore de Decisão")
plt.grid()
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.legend()
plt.show()