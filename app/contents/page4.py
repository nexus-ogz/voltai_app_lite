from contents import *
import plotly.express as px
import httpx
from ressources.models_fitted.random_forests.model_features import df_summary_with_mapping


URL_PRICING_KWH = "https://open-dpe.fr/api/v1/electricity.php?tarif=EDF_bleu"
PROD_DATA_FILE = "app/data/prod.json"
metadata_tarif = {
    'source': 'EDF - Tarif Bleu (API real time)',
    'description': 'Tarif reglementé - fixé par les pouvoirs publics.',
    'url': 'https://particulier.edf.fr/fr/accueil/electricite-gaz/tarif-bleu.html',
    'date_tarif': '01/02/2025',
    'date_extraction': '09/02/2025',
    'prix_kwh_base': 0.2016, # prix backp
    'prix_hc': 'NC',
    'prix_hp': 'NC'
}
try:
    query_pricing = httpx.get(URL_PRICING_KWH, timeout=5)
    if query_pricing.status_code == 200:
        res = query_pricing.json()
        metadata_tarif.update({
            'date_tarif': res.get('date_tarif', 'NA'),
            'date_extraction': res.get('date_extraction', 'NA'),
            'prix_kwh_base': res.get('options', {}).get('base', {}).get('prix_kWh', 0),
        })
except httpx.TimeoutException:
    print("Timeout occurred while fetching pricing data.")
except Exception as e:
    print(f"Error occurred while fetching pricing data: {e}")

PRIX_KWH_EUROS = metadata_tarif.get("prix_kwh_base")

def get_dpe_conso_range(inp):
    mapping = {
        "A": "inférieure à 70 kWh/m2/an",	
        "B": "entre 71 et 110 kWh/m2/an",	
        "C": "entre 111 et 180 kWh/m2/an",	
        "D": "entre 181 et 250 kWh/m2/an",	
        "E": "entre 251 et 330 kWh/m2/an",	
        "F": "entre 331 et 420 kWh/m2/an",	
        "G": "supérieure à 421 kWh/m2.an"
    }
    return mapping.get(inp, 'NC')

def load_hist_data():
    one = pd.read_json("app/data/train_df.json").sort_index(axis=1)
    # enlever les pred pareils - update caching avant predict 
    two = pd.read_json("app/data/prod.json", lines=True).sort_index(axis=1).drop_duplicates() 
    return pd.concat([one, two], axis=0)

model_metadata = {
    "path": "ressources/models_fitted/random_forests/random_forest_model_v1_restreint.pkl",
    "type": "Random Forest",
    "version": "rf_version_1_rest",
    "train_scope": None,
    "description": "Modèle de Random Forest pour la prédiction de la consommation électrique",
    "model_state": None,
    "features": df_summary_with_mapping
}

def get_dpe_label(dpe_value_idx):
    """
    Pour obtenir la liste des dpe restants. 
    Prédire les économies réalisées en changeant de classe DPE
    """
    mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G'}
    return mapping.get(dpe_value_idx, "NC")


def main(selected_ville, selected_annee, model_metadata=model_metadata):
    load_hist_data()
    st.subheader("Modélisation de la consommation électrique et des économies réalisables")

    loaded_model = load_pickle(model_metadata["path"]) #, type="pickle")
    model_state = ":green[loaded]" if loaded_model else ":red[error/not loaded]"

    not_show_metadata_model = st.toggle("Cacher les metadata du modèle")
    if not_show_metadata_model:
        pass
    else:
        st.markdown(f"""
    **Type :** *{model_metadata['type']}*\n
    **Path :** *{model_metadata['path']}*\n
    **Version :** *{model_metadata['version']}*\n
    **Train scope :** *{selected_ville}* *{selected_annee}*\n
    **Description :** *{model_metadata['description']}*\n
    **Model state :** *{model_state}*       
                    """)

    st.markdown("--------------------")
    if loaded_model:
        st.subheader("Application")
        model_features_config: dict = df_summary_with_mapping
        # st.write(model_features_config)
        # formulaire de saisie sur 3 colonnes à partir de model_fetures
        FLOAT_COLS, CATEG_COLS = [], []
        for feature, config in model_features_config.items():
            if 'float' in config.get('dtype'):
                FLOAT_COLS.append(feature)
            elif 'int' in config.get('dtype'):
                CATEG_COLS.append(feature)
            else:
                pass

        input_values = {}
        col11, col12 = st.columns(2)
        _milieu = len(FLOAT_COLS)//2
        for feature in FLOAT_COLS[:_milieu]:
            # default is min+max//2
            config = model_features_config.get(feature)
            label = config.get('desc', '')
            _default = float(config.get('def', 0))
            input_values[feature] = col11.slider(
                f"{label}",
                min_value=float(config.get('min', 0)),
                max_value=float(config.get('max', 100)),
                value=_default
            )
        for feature in FLOAT_COLS[_milieu:]:
            # default is min+max//2
            config = model_features_config.get(feature)
            label = config.get('desc', '')
            _default = float(config.get('def', 0))
            input_values[feature] = col12.slider(
                f"{label}",
                min_value=float(config.get('min', 0)),
                max_value=float(config.get('max', 100)),
                value=_default
            )
            
        col21, col22 = st.columns(2)
        _milieu = len(CATEG_COLS)//2
        for feature in CATEG_COLS[:_milieu]:
            config = model_features_config.get(feature)
            label = config.get('desc', '')
            v = col21.selectbox(f"{label}", options=config['mapping'], index=0)
            input_values[feature] = config.get('mapping').get(v)
        for feature in CATEG_COLS[_milieu:]:
            config = model_features_config.get(feature)
            label = config.get('desc', '')
            v = col22.selectbox(f"{label}", options=config['mapping'], index=0)
            input_values[feature] = config.get('mapping').get(v)

        input_dpe_value = input_values.get('etiquette_dpe_ademe')
        inputs_model = []
        for dpe in range(1, 8):
            input_values['etiquette_dpe_ademe'] = dpe
            inputs_model.append(input_values.copy())
        
        input_model_df = pd.DataFrame(inputs_model)
        input_model_df.sort_index(axis=1, inplace=True)

        # bouton de prédiction
        if st.button("Estimer votre consommation"):

            # 7 predictions - DPE A à G
            # prédire avec les mêmes inputs values les conso kwh/an/m2 et conso kwh/an
            # pour toutes les variantes du DPE 
            prediction = loaded_model.predict(input_model_df)

            rev_dpe_enc = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G"}
            # ajouter les predictions to the DataFrame
            input_model_df['conso_kwh_m2'] = prediction
            # sauv logs
            with open(PROD_DATA_FILE, 'a') as f:
                for record in input_model_df.to_dict(orient='records'):
                    f.write(json.dumps(record) + '\n')
                
            # refaire le df à afficher 
            res_df = input_model_df.copy()
            res_df['Conso kwh/m2/an'] = round(res_df['conso_kwh_m2'], 3)
            res_df['Conso kwh/an'] = round(res_df['Conso kwh/m2/an'] * res_df['surface_habitable_logement_ademe'], 3)
            res_df['Etiquette DPE'] = res_df['etiquette_dpe_ademe'].apply(lambda r: rev_dpe_enc.get(r))
            res_df['Montant (euros/an)'] = round(res_df['Conso kwh/an'] * PRIX_KWH_EUROS, 2)
            res_df['Economies (euros/an)'] = round(res_df['Montant (euros/an)'].shift(-1) - res_df['Montant (euros/an)'], 2)
            res_df = res_df.sort_values(by=['Etiquette DPE'], ascending=False)
            res_df['Economies cumulées (euros/an)'] = res_df['Economies (euros/an)'].cumsum()
            res_df = res_df.sort_values(by=['Etiquette DPE'], ascending=True)

            res_cols = [
                "Etiquette DPE",
                "Conso kwh/m2/an",
                "Conso kwh/an",
                "Montant (euros/an)",
                "Economies (euros/an)",
                "Economies cumulées (euros/an)"
            ]
            res_df = res_df[res_cols].rename(
                columns={
                    "surface_habitable_logement_ademe": "Surface du logement",
                    })
            
            st.success(f"""
                **Résultats :**\n
                "➡️ Etiquette DPE" : ***{get_dpe_label(input_dpe_value)}***\n
                "📊 Consommation estimée sur la base du DPE (min-max)" : ***{get_dpe_conso_range(get_dpe_label(input_dpe_value))}***\n
                "✅ Consommation kwh/an estimée": ***{round(prediction[input_dpe_value]*input_values.get("surface_habitable_logement_ademe"), 3)}***\n
                "✅ Consommation kwh/m2/an estimée": ***{round(prediction[input_dpe_value], 3)}***,\n
                "💶 Consommation euros/an estimée": ***{round(prediction[input_dpe_value] * PRIX_KWH_EUROS * input_values.get("surface_habitable_logement_ademe"), 3)}***,\n
                """)
            st.markdown("Informations sur la tarification appliquée")
            st.write(metadata_tarif)

            def highlight_row(s):
                if s['Etiquette DPE'] == get_dpe_label(input_dpe_value):
                    return ['background-color: #ffe082'] * len(s)
                else:
                    return [''] * len(s)

            def format_euro(val):
                try:
                    return f"{val:,.2f} €"
                except Exception:
                    return val

            styled_df = res_df.style.apply(highlight_row, axis=1)
            styled_df = styled_df.format({
                "Economies (euros/an)": format_euro,
                "Economies cumulées (euros/an)": format_euro,
                "Montant (euros/an)": format_euro,
                "Conso kwh/an": lambda v: f"{v:,.2f}",
                "Conso kwh/m2/an": lambda v: f"{v:,.2f}",
            })

            st.markdown("**Détails prédictions**")
            st.write("- *Montant euros/an = prix du kwh (euros/an) x consommation kwh/an*")
            st.write("- *Economie euros/an = Economie réalisée en changeant vers :orange[la classe DPE au dessus]*")
            st.write("- *Economie cumulée euros/an = Economie réalisée en changeant de :orange[plusieurs classes DPE] au dessus*")

            st.dataframe(
                styled_df,
                hide_index=True
            )


    # ----------------------------------------
    # recyclage
    # logguer les data dans un dossier de app

    # concept drift 
    # conso 
    # plot distribution de départ vs distrib (données de départ + new)

    # data drift 
    # plot la distrib avant des variables
    # plot l'apres 
    # -----------------------------------------

    monitoring_enabled = st.toggle("Display - drift Monitoring")

    if not monitoring_enabled: 
        st.warning('Monitoring disabled !')
    else: 
        train_data = pd.read_json("app/data/train_df.json")
        hist_data = load_hist_data() # load + add prod data

        target = 'conso_kwh_m2'
        features = [c for c in train_data.columns if c !=target]
        d1, d2 = st.columns(2)

        ###### target col
        d1.markdown("Concept drift - Target distribution")
        _ = d1.selectbox(label="target", options=[target])
        target_drift_train = np.log(train_data[[target]].copy())
        target_drift_hist = np.log(hist_data[[target]].copy())

        # column to distinguish datasets
        target_drift_train['dataset'] = 'train'
        target_drift_hist['dataset'] = 'hist'

        target_drift_all = pd.concat([target_drift_train, target_drift_hist], axis=0)
        target_drift_all = target_drift_all.query(f"{target} > 0")
        fig = px.histogram(
            target_drift_all,
            x=target,
            color='dataset',
            histnorm='probability density',
            title=f"Probability Density of {target} - log scale",
            # nbins=50,
            barmode='overlay',
            color_discrete_map={'train': 'blue', 'hist': 'orange'}
        )
        fig.update_traces(opacity=0.6)
        fig.update_layout(yaxis_title="Probability Density")
        d1.plotly_chart(fig, use_container_width=True)


        ###### features 
        d2.markdown("Features drift")
        drift_feature_selected = d2.selectbox(label="select feature", options=features)
        feature_drift_df = train_data[drift_feature_selected]

        if 'float' in model_features_config.get(drift_feature_selected, {}).get('dtype'):
            feature_drift_train = np.log(train_data[[drift_feature_selected]].copy())
            feature_drift_hist = np.log(hist_data[[drift_feature_selected]].copy())

            # column to distinguish datasets
            feature_drift_train['dataset'] = 'train'
            feature_drift_hist['dataset'] = 'hist'

            feature_drift_all = pd.concat([feature_drift_train, feature_drift_hist], axis=0)
            feature_drift_all = feature_drift_all.query(f"{drift_feature_selected}>0")
            fig = px.histogram(
                feature_drift_all,
                x=drift_feature_selected,
                color='dataset',
                histnorm='probability density',
                title=f"Probability Density - log scale",
                # nbins=50,
                barmode='overlay',
                color_discrete_map={'train': 'orange', 'hist': 'blue'}
            )
            fig.update_traces(opacity=0.6)
            fig.update_layout(yaxis_title="Probability Density")
            d2.plotly_chart(fig, use_container_width=True)

        if 'int' in model_features_config.get(drift_feature_selected, {}).get('dtype'):
            
            d2.markdown("")
            d2.markdown("")
            d2.markdown("**Probability distribution - categ. feature**")
            d2.markdown("")

            feature_train_data = train_data[[drift_feature_selected]].copy().value_counts(normalize=True).reset_index().set_index(drift_feature_selected)
            feature_hist_data = hist_data[[drift_feature_selected]].copy().value_counts(normalize=True).reset_index().set_index(drift_feature_selected)
            feature_train_data = feature_train_data.rename(columns={'proportion': 'training_proportions'})
            feature_hist_data = feature_hist_data.rename(columns={'proportion': 'historic_proportions'})

            feature_all_data = pd.merge(feature_hist_data, feature_train_data, left_index=True, right_index=True)

            m_ = model_features_config.get(drift_feature_selected, {}).get('mapping', {})

            d2.dataframe(feature_all_data)
            d2.write(m_)
