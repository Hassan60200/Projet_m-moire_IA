import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

st.title("Projet de prédiction immobilière")
st.write("Cette application est un outil de prédiction des prix immobiliers.")


@st.cache_data
def load_data():
    dvf = pd.read_csv('./data/dvf_compiegne.csv', delimiter=';')
    dvf = dvf[['nature_mutation', 'valeur_fonciere', 'date_mutation', 'type_local', 'nombre_pieces_principales',
               'surface_reelle_bati', 'code_departement']]

    chomage = pd.read_excel('./data/chomage.xlsx')
    chomage = chomage.dropna()
    chomage = chomage.drop(chomage.index[:1])
    chomage.rename(columns={'Observatoire des territoires - ANCT': 'Code'}, inplace=True)
    chomage.rename(columns={'Unnamed: 1': 'Departements'}, inplace=True)
    chomage['Departements'] = chomage['Departements'].drop(chomage.index[-1])
    chomage.rename(columns={'Unnamed: 2': 'Année'}, inplace=True)
    chomage.rename(columns={'Unnamed: 3': 'Taux de chômage'}, inplace=True)

    logement = pd.read_excel('./data/logement_sociaux_2022.xlsx', sheet_name='DEP', engine='openpyxl')
    logement = logement.drop(logement.index[:3])

    new_column_names = {'Logements sociaux au 1er janvier 2022 : comparaisons départementales': 'Code',
                        'Unnamed: 1': 'Départements', 'Unnamed: 2': 'Nbre de logements pour 10 000 habitants',
                        'Unnamed: 5': 'Loyer moyen par m²'}
    columns_to_drop = ['Unnamed: 3', 'Unnamed: 4']

    logement = logement.drop(columns=columns_to_drop)
    logement = logement.rename(columns=new_column_names)

    dvf = dvf.drop_duplicates()
    chomage = chomage.drop_duplicates()
    logement = logement.drop_duplicates()

    dvf['annee'] = pd.to_datetime(dvf['date_mutation']).dt.year
    chomage['Année'] = chomage['Année'].astype(int)

    dvf['code_departement'] = dvf['code_departement'].astype(str)
    chomage.rename(columns={'Code': 'code_departement'}, inplace=True)
    chomage['code_departement'] = chomage['code_departement'].astype(str)

    chomage = chomage[chomage['code_departement'] == '60']
    logement = logement[logement['Départements'] == 'Oise']
    logement = logement[logement['Code'] == '60']

    dvf = pd.merge(dvf, chomage, how='left', left_on=['code_departement', 'annee'],
                   right_on=['code_departement', 'Année'])
    dvf = pd.merge(dvf, logement, how='left', left_on='code_departement', right_on='Code')
    dvf['Departements'] = dvf['Departements'].fillna('Oise')
    dvf['Taux de chômage'] = dvf['Taux de chômage'].fillna(7.3)
    dvf['nombre_pieces_principales'] = dvf['nombre_pieces_principales'].fillna(
        dvf['nombre_pieces_principales'].median())
    dvf['surface_reelle_bati'] = dvf['surface_reelle_bati'].fillna(dvf['surface_reelle_bati'].median())

    dvf.drop(columns=['Année', 'Départements', 'Code'], inplace=True)

    return dvf, chomage, logement


dvf, chomage, logement = load_data()

st.write("Taux de chômage :")
st.dataframe(chomage)

st.write("Logements sociaux :")
st.dataframe(logement)

st.write("Demande valeur foncière :")
st.dataframe(dvf)

df_annual = dvf.groupby('annee').mean()

plt.figure(figsize=(10, 6))
plt.plot(df_annual.index, df_annual['valeur_fonciere'])
plt.title('Valeur foncière moyenne par an')
plt.xlabel('Année')
plt.ylabel('Valeur foncière moyenne')
plt.show()

st.pyplot(plt.gcf())

# Créer un formulaire
with st.form(key='my_form'):
    st.write("Remplissez le formulaire :")
    type_local_input = st.selectbox(label='Type de bien immobilier', options=['Appartements', 'Maison', 'Autres'])
    pieces_input = st.number_input(label='Nombre de pièces principales', min_value=1, max_value=10, value=5)
    surface_input = st.number_input(label='Surface réelle bâtie (m²)', min_value=1.0, max_value=500.0, value=100.0)
    chomage_input = st.number_input(label='Taux de chômage (%)', min_value=1.0, max_value=50.0, value=7.3)
    logements_input = st.number_input(label='Nombre de logements pour 10 000 habitants', min_value=1, max_value=10000, value=821)
    loyer_input = st.number_input(label='Loyer moyen par m² (€)', min_value=1.0, max_value=50.0, value=5.69)
    submit_button = st.form_submit_button(label='Submit')

# Traitement des données du formulaire
if submit_button:
    dvf = dvf.append({'type_local': type_local_input,
                      'nombre_pieces_principales': pieces_input,
                      'surface_reelle_bati': surface_input,
                      'Taux de chômage': chomage_input,
                      'Nbre de logements pour 10 000 habitants': logements_input,
                      'Loyer moyen par m²': loyer_input}, ignore_index=True)
    st.write("Les données du formulaire ont été ajoutées à dvf.")
    st.dataframe(dvf.tail())
