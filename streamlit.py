import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

st.title("Projet de prédiction immobilière")
st.write("Cette application est un outil de prédiction des prix immobiliers.")


@st.cache_data
@st.cache_resource
def load_dvf_data():
    dvf = pd.read_csv('./data/dvf_compiegne.csv', delimiter=';')
    dvf = dvf[['nature_mutation', 'valeur_fonciere', 'date_mutation', 'type_local', 'nombre_pieces_principales',
               'surface_reelle_bati', 'code_departement', 'surface_terrain', 'nombre_lots']]
    dvf['annee'] = pd.to_datetime(dvf['date_mutation']).dt.year
    dvf['code_departement'] = dvf['code_departement'].astype(str)
    dvf = dvf.drop_duplicates()
    dvf['surface_terrain'] = dvf['surface_terrain'].fillna(0)

    return dvf


@st.cache_resource
def load_chomage_data():
    df_chomage = pd.read_excel('./data/chomage.xlsx')
    df_chomage = df_chomage.dropna()
    df_chomage = df_chomage.drop(df_chomage.index[:1])
    df_chomage.rename(columns={'Observatoire des territoires - ANCT': 'Code'}, inplace=True)
    df_chomage.rename(columns={'Unnamed: 1': 'Departements'}, inplace=True)
    df_chomage['Departements'] = df_chomage['Departements'].drop(df_chomage.index[-1])
    df_chomage.rename(columns={'Unnamed: 2': 'Année'}, inplace=True)
    df_chomage.rename(columns={'Unnamed: 3': 'Taux de chômage'}, inplace=True)
    df_chomage['Année'] = df_chomage['Année'].astype(int)
    df_chomage.rename(columns={'Code': 'code_departement'}, inplace=True)
    df_chomage['code_departement'] = df_chomage['code_departement'].astype(str)
    df_chomage = df_chomage.drop_duplicates()
    return df_chomage


@st.cache_resource
def load_logement_data():
    df_logement = pd.read_excel('./data/logement_sociaux_2022.xlsx', sheet_name='DEP', engine='openpyxl')
    df_logement = df_logement.drop(df_logement.index[:3])
    new_column_names = {'Logements sociaux au 1er janvier 2022 : comparaisons départementales': 'Code',
                        'Unnamed: 1': 'Départements', 'Unnamed: 2': 'Nbre de logements pour 10 000 habitants',
                        'Unnamed: 5': 'Loyer moyen par m²'}
    columns_to_drop = ['Unnamed: 3', 'Unnamed: 4']
    df_logement = df_logement.drop(columns=columns_to_drop)
    df_logement = df_logement.rename(columns=new_column_names)
    df_logement = df_logement.drop_duplicates()
    return df_logement


def merge_data():
    dvf = load_dvf_data()
    df_chomage = load_chomage_data()
    df_logement = load_logement_data()

    dvf = pd.merge(dvf, df_chomage, how='left', left_on=['code_departement', 'annee'],
                   right_on=['code_departement', 'Année'])
    dvf = pd.merge(dvf, df_logement, how='left', left_on='code_departement', right_on='Code')

    dvf['Departements'] = dvf['Departements'].fillna('Oise')
    dvf['Taux de chômage'] = dvf['Taux de chômage'].fillna(7.3)
    dvf['nombre_pieces_principales'] = dvf['nombre_pieces_principales'].fillna(
        dvf['nombre_pieces_principales'].median())
    dvf['surface_reelle_bati'] = dvf['surface_reelle_bati'].fillna(dvf['surface_reelle_bati'].median())

    dvf.drop(columns=['Année', 'Départements', 'Code'], inplace=True)
    dvf['nature_mutation'] = dvf['nature_mutation'].map({'Vente': 1, 'Echange': 0})
    dvf['type_local'] = dvf['type_local'].replace(['Local industriel. commercial ou assimilé'], 'Autres')

    dvf['date_mutation'] = pd.to_datetime(dvf['date_mutation'])
    dvf['annee_mutation'] = dvf['date_mutation'].dt.year
    dvf['mois_mutation'] = dvf['date_mutation'].dt.month
    dvf['jour_mutation'] = dvf['date_mutation'].dt.day

    dvf = dvf.drop(columns=['date_mutation'])
    dvf = dvf[dvf['type_local'].isin(['Appartement', 'Maison', 'Autres'])]
    dvf = pd.get_dummies(dvf, columns=['type_local'])

    return dvf


def display_data_Compègne(dvf, df_chomage, df_logement):
    st.write("Taux de chômage :")
    st.dataframe(df_chomage)

    st.write("Logements sociaux :")
    st.dataframe(df_logement)

    st.write("Demande valeur foncière Compiègne:")
    st.dataframe(dvf)

    df_annual = dvf.groupby('annee').mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df_annual.index, df_annual['valeur_fonciere'])
    plt.title('Valeur foncière moyenne par an')
    plt.xlabel('Année')
    plt.ylabel('Valeur foncière moyenne')
    plt.show()

    st.pyplot(plt.gcf())


dvf = merge_data()
df_chomage = load_chomage_data()
df_logement = load_logement_data()

display_data_Compègne(dvf, df_chomage, df_logement)

unique_departments = df_chomage['Departements'].unique().tolist()
# Créer un formulaire
with st.form(key='my_form'):
    model_option = st.selectbox("Choisissez un modèle",
                                ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor"])
    dept_input = st.selectbox('Sélectionnez le département', [""] + unique_departments)
    nature_mutation_input = st.selectbox(label='Nature mutation', options=['Vente', 'Echange'])
    type_local_input = st.selectbox(label='Type de bien immobilier', options=['Appartement', 'Maison', 'Autres'])
    pieces_input = st.number_input(label='Nombre de pièces principales', min_value=1, max_value=10, value=5)
    surface_input = st.number_input(label='Surface réelle bâtie (m²)', min_value=1.0, max_value=500.0, value=100.0)
    terrain_input = st.number_input(label='Surface du terrain (m²)', min_value=1.0, max_value=10000.0, value=500.0)
    lots_input = st.number_input(label='Nombre de lots', min_value=1, max_value=10, value=1)
    chomage_input = st.number_input(label='Taux de chômage (%)', min_value=1.0, max_value=50.0)
    logements_input = st.number_input(label='Nombre de logements pour 10 000 habitants', min_value=1, max_value=10000)
    loyer_input = st.number_input(label='Loyer moyen par m² (€)', min_value=1.0, max_value=50.0)
    type_local_appartement = 1 if type_local_input == 'Appartement' else 0
    type_local_autres = 1 if type_local_input == 'Autres' else 0
    type_local_maison = 1 if type_local_input == 'Maison' else 0
    nature_mutation_input = 1 if nature_mutation_input == "Vente" else 0
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    form_data = pd.DataFrame({
        'nature_mutation': [nature_mutation_input],
        'nombre_pieces_principales': [pieces_input],
        'surface_reelle_bati': [surface_input],
        'code_departement': [60],
        'surface_terrain': [terrain_input],
        'nombre_lots': [lots_input],
        'annee': [2022],
        'Taux de chômage': [chomage_input],
        'Nbre de logements pour 10 000 habitants': [logements_input],
        'Loyer moyen par m²': [loyer_input],
        'annee_mutation': [2023],
        'mois_mutation': [7],
        'jour_mutation': [15],
        'type_local_Appartement': [type_local_appartement],
        'type_local_Autres': [type_local_autres],
        'type_local_Maison': [type_local_maison],
    })

    X = dvf.drop(columns=['valeur_fonciere', 'Departements'])
    y = dvf['valeur_fonciere']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    if model_option == "LinearRegression":
        model = LinearRegression()
    elif model_option == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(random_state=42)
    elif model_option == "RandomForestRegressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    prediction = model.predict(form_data)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f'La valeur foncière prédite est : {round(prediction[0], 2)}' '€')

    st.write(f'Le score R2 est :  {round(r2, 2)}')
