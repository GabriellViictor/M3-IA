import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Carregar os dados de treinamento e teste
data_train = pd.read_csv('bank/bank.csv', delimiter=',')
data_test = pd.read_csv('bank/bank-full.csv', delimiter=',')

# Pré-processamento dos dados
# 1. Transformar variáveis categóricas em dummies
data_train = pd.get_dummies(data_train, drop_first=True)
data_test = pd.get_dummies(data_test, drop_first=True)

data_test.head()
data_train.head()
# Garantir que os conjuntos de treino e teste tenham as mesmas colunas
data_test = data_test.reindex(columns=data_train.columns, fill_value=0)

data_test.head()

# Separar características (X) e rótulo/target (y)
target_column = "y_yes"
X_train = data_train.drop(target_column, axis=1)
y_train = data_train[target_column]
X_test = data_test.drop(target_column, axis=1)
y_test = data_test[target_column]

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir e treinar a rede neural
mlp = MLPClassifier(hidden_layer_sizes=(30, 20, 10), max_iter=10000, tol=1e-9, random_state=42)
mlp.fit(X_train, y_train)

# Testar a rede neural no conjunto de teste
y_pred = mlp.predict(X_test)

# Calcular e imprimir a matriz de confusão e as métricas de avaliação
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label=True)
precision = precision_score(y_test, y_pred, pos_label=True)
recall = recall_score(y_test, y_pred, pos_label=True)

print(len(data_test.keys()))

print("Matriz de Confusão (Conjunto de Teste):")
print(conf_matrix)
print(f"Acurácia: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
