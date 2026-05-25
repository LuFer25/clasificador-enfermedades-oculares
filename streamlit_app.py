import streamlit as st
import numpy as np
import cv2
import joblib
import pandas as pd

# =========================
# CONFIGURACIÓN PÁGINA
# =========================
st.set_page_config(
    page_title="Clasificador de Enfermedades Oculares",
    page_icon="👁️",
    layout="wide"
)

# =========================
# PALETA Y CSS
# =========================
try:
    with open("styles.css", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
except FileNotFoundError:
    st.warning("No se encontró el archivo styles.css, usando estilos por defecto.")

# =========================
# CONSTANTES Y DICCIONARIOS
# =========================
TAMAÑO = 128

classes = {
    0: 'Glaucoma',
    1: 'Normal',
    2: 'Retinopatía Diabética',
    3: 'Cataratas'
}

# =========================
# CARGA DE LOS 4 PIPELINES
# =========================
@st.cache_resource
def load_all_pipelines():
    try:
        pipe_rf = joblib.load('pipeline_RF.sav')
        pipe_xgb = joblib.load('pipeline_XGBoost.sav')   
        pipe_st = joblib.load('pipeline_stacking.sav')   
        pipe_mlp = joblib.load('pipeline_MLP.sav') 
        return pipe_rf, pipe_xgb, pipe_st, pipe_mlp
    except FileNotFoundError as e:
        st.error(f"Error: No se pudo encontrar uno de los archivos de pipeline. Detalles: {e}")
        return None, None, None, None

pipe_rf, pipe_xgb, pipe_st, pipe_mlp = load_all_pipelines()

def predict_with_pipeline(pipeline, image_input):
    pred_numeric = pipeline.predict(image_input)[0]
    probs = pipeline.predict_proba(image_input)
    
    # SciKeras/Keras a veces devuelve las probabilidades envueltas en dimensiones extra
    # o devuelve un escalar. Aquí nos aseguramos de aplanarlo a un array de 1D.
    probs = np.squeeze(probs) 
    
    pred_name = classes.get(pred_numeric, "Desconocido")
    
    # Verificación de seguridad: si probs sigue siendo un número suelto (escalar) y no un array
    if probs.ndim == 0: 
        # Si es un escalar, asumimos que representa la confianza de la clase predicha
        confidence = float(probs)
        # Reconstruimos un array ficticio para que las tablas de probabilidad no se rompan
        probs_mismo_tamaño = [0.0] * len(classes)
        if pred_numeric < len(probs_mismo_tamaño):
            probs_mismo_tamaño[pred_numeric] = confidence
        probs = np.array(probs_mismo_tamaño)
    else:
        # Si es un array normal (como RF o XGBoost), extraemos la confianza normalmente
        confidence = probs[pred_numeric]
        
    return pred_name, confidence, probs

# =========================
# HEADER
# =========================
st.markdown("""
<div class="main-card">
    <p style="text-align:center; font-size:20px; color:#F4F0FD;">
        UNIVERSIDAD AUTÓNOMA DE CHIHUAHUA
        Facultad de Ingeniería
    </p>
    <h1>👁️ Clasificador de Enfermedades Oculares</h1>
    <p style="text-align:center; font-size:20px; color:#F4F0FD;">
        Materia: Data Science

Asesor: Olanda Prieto Ordaz

Nombre del alumno: Luisa Fernanda Hernández Hernández

Matrícula: 368068
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# COMPONENTE DE CARGA
# =========================
if pipe_rf is not None and pipe_xgb is not None and pipe_st is not None and pipe_mlp is not None:
    uploaded_file = st.file_uploader(
        "📤 Sube una imagen retinal",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col_img, col_results = st.columns([1, 2])

        with col_img:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(
                img_rgb,
                caption="Imagen retinal subida",
                use_container_width=True
            )

        # =========================
        # PREPROCESAMIENTO EXACTO
        # =========================
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (TAMAÑO, TAMAÑO))
        img_flattened = img_resized.flatten()
        image_for_prediction = img_flattened.reshape(1, -1)

        # =========================
        # PREDICCIONES INDEPENDIENTES
        # =========================
        # Función auxiliar modificada para proteger la predicción del MLP

        rf_pred, rf_conf, rf_probs = predict_with_pipeline(pipe_rf, image_for_prediction)
        xgb_pred, xgb_conf, xgb_probs = predict_with_pipeline(pipe_xgb, image_for_prediction)
        st_pred, st_conf, st_probs = predict_with_pipeline(pipe_st, image_for_prediction)
        mlp_pred, mlp_conf, mlp_probs = predict_with_pipeline(pipe_mlp, image_for_prediction)

        # =========================
        # MOSTRAR RESULTADOS GENERALES
        # =========================
        with col_results:
            st.subheader("Resultados Generales")
            
            resumen_df = pd.DataFrame({
                "Modelo": ["Random Forest", "XGBoost", "Stacking Classifier", "MLP"],
                "Clase Predicha": [rf_pred, xgb_pred, st_pred, mlp_pred],
                "Probabilidad": [f"{rf_conf:.2%}", f"{xgb_conf:.2%}", f"{st_conf:.2%}", f"{mlp_conf:.2%}"]
            })
            
            st.dataframe(resumen_df, use_container_width=True, hide_index=True)

        st.write("")
        st.subheader("Probabilidad Detallada por Enfermedad")
        
        prob_df = pd.DataFrame({
            "Enfermedad": list(classes.values()),
            "Random Forest": [f"{p:.2%}" for p in rf_probs],
            "XGBoost": [f"{p:.2%}" for p in xgb_probs],
            "Stacking Classifier": [f"{p:.2%}" for p in st_probs],
            "MLP": [f"{p:.2%}" for p in mlp_probs] # <--- Probabilidades Modelo 4
        })
        
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

        # =========================
        # TARJETAS FINALES COMPARATIVAS
        # =========================
        st.write("")
        st.subheader("Predicción Final de Cada Modelo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title">Random Forest</div>
                <div class="prediction-class">{rf_pred}</div>
                <div class="prediction-prob">{rf_conf:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title">XGBoost</div>
                <div class="prediction-class">{xgb_pred}</div>
                <div class="prediction-prob">{xgb_conf:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title">Stacking</div>
                <div class="prediction-class">{st_pred}</div>
                <div class="prediction-prob">{st_conf:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4: 
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title">Cuarto Modelo</div>
                <div class="prediction-class">{mlp_pred}</div>
                <div class="prediction-prob">{mlp_conf:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.warning("Por favor, asegúrate de que los archivos 'pipeline_RF.sav', 'pipeline_XGBoost.sav', 'pipeline_stacking.sav' y 'pipeline_MLP.sav' estén en la misma carpeta.")