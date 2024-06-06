import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def load_nhanes_data():
    urls = {
        'demo': 'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DEMO_F.XPT',
        'lab': 'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/GLU_F.XPT',
        'body': 'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/BMX_F.XPT',
        'bp': 'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/BPX_F.XPT',
        'questionnaire': 'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DIQ_F.XPT'
    }

    try:
        demo_data = pd.read_sas(urls['demo'])
        lab_data = pd.read_sas(urls['lab'])
        body_data = pd.read_sas(urls['body'])
        bp_data = pd.read_sas(urls['bp'])
        questionnaire_data = pd.read_sas(urls['questionnaire'])
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

    # Merge datasets on SEQN (unique identifier for each participant)
    data = demo_data.merge(lab_data, on='SEQN').merge(body_data, on='SEQN').merge(bp_data, on='SEQN').merge(questionnaire_data, on='SEQN')

    return data

def preprocess_data(data):
    columns = ['RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'DMDEDUC2', 'DMDMARTL', 
               'BPXSY1', 'BPXDI1', 'BMXBMI', 'LBXGLU', 'DIQ010']
    
    # Verificar se todas as colunas estão presentes
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        print(f"Colunas ausentes nos dados: {missing_columns}")
        # Remover as colunas ausentes da lista de colunas
        columns = [col for col in columns if col in data.columns]
        # Se a coluna 'DIQ010' estiver ausente, tratar de outra forma
        if 'DIQ010' in missing_columns:
            print("A coluna 'DIQ010' está ausente. A classificação de diabetes não será possível.")
            return None, None

    data = data[columns]

    # Definir novos nomes para as colunas existentes
    new_column_names = ['Age', 'Gender', 'Ethnicity', 'Education', 'MaritalStatus', 
                        'SystolicBP', 'DiastolicBP', 'BMI', 'Glucose', 'Diabetes']
    
    # Ajustar o número de novos nomes de colunas conforme o número de colunas existentes
    new_column_names = new_column_names[:len(data.columns)]

    # Renomear colunas
    data.columns = new_column_names

    # Remover entradas com valores ausentes
    data = data.dropna()

    # Transformar variáveis categóricas em variáveis dummy
    data = pd.get_dummies(data, columns=['Gender', 'Ethnicity', 'Education', 'MaritalStatus'], drop_first=True)

    # Definir X e y
    X = data.drop('Diabetes', axis=1)
    y = data['Diabetes'].apply(lambda x: 1 if x == 1 else 0)  # Transformar 1 (sim) em 1 e outros em 0

    return X, y

def build_and_train_model(X_train, y_train, X_test, y_test):
    # Construir o Modelo da Rede Neural
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.5),  # Adicionando dropout para evitar overfitting
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compilar o Modelo
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Treinar o Modelo
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, 
                        validation_data=(X_test, y_test))

    return model, history

def evaluate_model(model, X_test, y_test):
    # Avaliar o Modelo
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nAcurácia do modelo no conjunto de teste: {test_acc}')

    # Fazer previsões e gerar relatório de classificação
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))

def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perda no Treinamento')
    plt.plot(history.history['val_loss'], label='Perda na Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.title('Perda ao longo das Épocas')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia no Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia na Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.title('Acurácia ao longo das Épocas')

    plt.show()

# Carregar e pré-processar os dados
data = load_nhanes_data()
if data is not None:
    X, y = preprocess_data(data)
    if X is not None and y is not None:
        # Dividir os dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Padronizar os dados
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Construir, treinar e avaliar o modelo
        model, history = build_and_train_model(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test)

        # Visualizar a perda e a acurácia ao longo das épocas
        plot_history(history)
    else:
        print("Pré-processamento falhou devido a colunas ausentes.")
else:
    print("Falha ao carregar os dados. Verifique as URLs e tente novamente.")
