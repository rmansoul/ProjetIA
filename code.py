import streamlit as st
import pandas as pd
import numpy as np

def prepare_X(df, base):
    df_num = df[base]
    df_num = df_num.fillna(df_num.mean())
    return df_num.values

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.pinv(XTX)  # Utilisation de la pseudo-inverse
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]

# Charger les données
df = pd.read_csv("data1.csv")
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.rename(columns={'msrp': 'price'}, inplace=True)
df['log_price'] = np.log1p(df.price)

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
X_train = prepare_X(df, base)
y_train = df.log_price.values
w_0, w = train_linear_regression(X_train, y_train)

# Interface Streamlit
st.title("Prédiction du Prix des Voitures")
st.write("Entrez les caractéristiques de votre voiture pour obtenir une estimation de son prix.")

# Champs de saisie utilisateur
engine_hp = st.number_input("Puissance du moteur (HP)", value=268.0)
engine_cylinders = st.number_input("Nombre de cylindres", value=1.0)
highway_mpg = st.number_input("Consommation autoroute (MPG)", value=5.0)
city_mpg = st.number_input("Consommation ville (MPG)", value=2.0)
popularity = st.number_input("Popularité", value=2031)

# Bouton de prédiction
if st.button("Prédire le prix"):
    ad = {
        'engine_hp': engine_hp,
        'engine_cylinders': engine_cylinders,
        'highway_mpg': highway_mpg,
        'city_mpg': city_mpg,
        'popularity': popularity
    }
    X_test = prepare_X(pd.DataFrame([ad]), base)
    y_pred = w_0 + X_test.dot(w)
    price = np.expm1(y_pred)
    
    st.success(f"Prix estimé: ${price[0]:,.2f}")
