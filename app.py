import streamlit as st
import joblib
import numpy as np
import re
import unicodedata
import nltk

# Descargar recursos de NLTK (solo la primera vez)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords


STOP_ES = frozenset(stopwords.words("spanish"))

NOMBRES_ODS = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos",
}

CORTE_ALTA, CORTE_BAJA = 0.30, 0.15



def text_preprocess(text):
    """Misma función de preprocesamiento del notebook."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = "".join(
        c if c in ("ñ", "Ñ")
        else unicodedata.normalize("NFD", c)[0]
        if unicodedata.category(unicodedata.normalize("NFD", c)[-1]) == "Mn"
        and len(unicodedata.normalize("NFD", c)) > 1
        else c
        for c in text
    )
    text = re.sub(r"[^a-záéíóúñü\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_ES and len(t) > 2]
    return " ".join(tokens)


def puntajes(modelo, textos):
    """Softmax sobre decision_function para obtener pseudo-probabilidades."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(textos)
    decision = modelo.decision_function(textos)
    exp = np.exp(decision - decision.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def nivel_confianza(p):
    if p > CORTE_ALTA:
        return " Alta"
    elif p > CORTE_BAJA:
        return " Media"
    return " Baja"


def etiqueta_ods(k):
    return f"ODS {k}: {NOMBRES_ODS.get(k, '?')}"



@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_ods.joblib")


modelo = cargar_modelo()

st.set_page_config(page_title="Clasificador ODS", page_icon="🌍", layout="centered")
st.title("🌍 Clasificador de Objetivos de Desarrollo Sostenible")
st.markdown(
    "Ingresa un texto en español y el modelo predecirá a cuál ODS corresponde. "
    "Se muestran los **3 ODS más probables** con su nivel de confianza."
)

texto_usuario = st.text_area(
    "Escribe o pega tu texto aquí:",
    height=150,
    placeholder="Ejemplo: El municipio invertirá en plantas de tratamiento de aguas residuales para garantizar agua potable segura en las zonas rurales.",
)

if st.button(" Clasificar", type="primary"):
    if not texto_usuario.strip():
        st.warning("Por favor, ingresa un texto para clasificar.")
    else:
        with st.spinner("Clasificando..."):
            probs = puntajes(modelo, [texto_usuario])[0]
            clases = modelo.classes_
            top3_idx = np.argsort(probs)[::-1][:3]

        # Resultado principal
        pred_idx = top3_idx[0]
        pred_clase = clases[pred_idx]
        pred_prob = probs[pred_idx]
        confianza = nivel_confianza(pred_prob)

        st.markdown("---")
        st.subheader(f"Predicción: {etiqueta_ods(pred_clase)}")
        st.markdown(f"**Confianza:** {confianza} ({pred_prob:.1%})")

        if pred_prob <= CORTE_BAJA:
            st.warning(
                "Confianza baja: esta predicción acierta ~68% de las veces. "
                "Revisa las alternativas."
            )

        # Top 3
        st.markdown("#### Top 3 ODS más probables")
        for rank, idx in enumerate(top3_idx, 1):
            clase = clases[idx]
            prob = probs[idx]
            st.progress(prob, text=f"{rank}. {etiqueta_ods(clase)} — {prob:.1%}")

st.markdown("---")
st.caption("Modelo: LinearSVC + TF-IDF + SVD (300 componentes) · Reto 2  Aprendizaje No Supervisado")