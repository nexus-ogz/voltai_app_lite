from contents import *
from contents import page1, page4

# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_page_config(layout="wide")

pages = {
    " Présentation générale": page1.main,
    " Démo": page4.main,
}

st.sidebar.title('Navigation')
p = st.sidebar.radio('Aller à  ', list(pages.keys()))

st.sidebar.markdown("-------------------")

st.sidebar.header('Filtres')
ville_filter = st.sidebar.selectbox('Ville', ['Paris'])
annee_filter = st.sidebar.selectbox('Année', ['2022'])

pages[p](ville_filter, annee_filter)