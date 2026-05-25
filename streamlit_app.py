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

# Mapeo oficial de tu entrenamiento
classes = {
    0: 'Glaucoma',
    1: 'Normal',
    2: 'Retinopatía Diabética',
    3: 'Cataratas'
}

# =========================
# CARGA DE LOS 3 PIPELINES
# =========================
@st.cache_resource
def load_all_pipelines():
    try:
        # Cargamos tus 3 archivos de pipeline independientes
        pipe_rf = joblib.load('RF.sav')
        pipe_xgb = joblib.load('XGBoost.sav')   # <--- Agregado
        pipe_st = joblib.load('pipeline_stacking.sav')   # <--- Agregado
        return pipe_rf, pipe_xgb, pipe_st
    except FileNotFoundError as e:
        st.error(f"Error: No se pudo encontrar uno de los archivos de pipeline. Detalles: {e}")
        return None, None, None

pipe_rf, pipe_xgb, pipe_st = load_all_pipelines()

# Función auxiliar para predecir de forma segura con cada pipeline
def predict_with_pipeline(pipeline, image_input):
    pred_numeric = pipeline.predict(image_input)[0]
    probs = pipeline.predict_proba(image_input)[0]
    
    pred_name = classes.get(pred_numeric, "Desconocido")
    confidence = probs[pred_numeric]
    return pred_name, confidence, probs

# =========================
# HEADER
# =========================
st.markdown("""
<div class="main-card">
    <h1>👁️ Clasificador de Enfermedades Oculares</h1>
    <p style="text-align:center; font-size:20px; color:#F4F0FD;">
        Sube una imagen retinal y compara los resultados obtenidos simultáneamente con tus 3 modelos entrenados.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# COMPONENTE DE CARGA
# =========================
if pipe_rf is not None and pipe_xgb is not None and pipe_st is not None:
    uploaded_file = st.file_uploader(
        "📤 Sube una imagen retinal",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Convertir archivo subido a formato OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Diseño en dos columnas: Imagen vs Resultados Generales
        col_img, col_results = st.columns([1, 2])

        with col_img:
            # Mostramos la imagen usando el canal correcto para Streamlit (RGB)
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
        rf_pred, rf_conf, rf_probs = predict_with_pipeline(pipe_rf, image_for_prediction)
        xgb_pred, xgb_conf, xgb_probs = predict_with_pipeline(pipe_xgb, image_for_prediction)
        st_pred, st_conf, st_probs = predict_with_pipeline(pipe_st, image_for_prediction)

        # =========================
        # MOSTRAR RESULTADOS GENERALES
        # =========================
        with col_results:
            st.subheader("Resultados Generales")
            
            resumen_df = pd.DataFrame({
                "Modelo": ["Random Forest", "XGBoost", "Stacking Classifier"],
                "Clase Predicha": [rf_pred, xgb_pred, st_pred],
                "Probabilidad": [f"{rf_conf:.2%}", f"{xgb_conf:.2%}", f"{st_conf:.2%}"]
            })
            
            st.dataframe(resumen_df, use_container_width=True, hide_index=True)

        # Matriz completa de probabilidades por enfermedad
        st.write("")
        st.subheader("Probabilidad Detallada por Enfermedad")
        
        prob_df = pd.DataFrame({
            "Enfermedad": list(classes.values()),
            "Random Forest": [f"{p:.2%}" for p in rf_probs],
            "XGBoost": [f"{p:.2%}" for p in xgb_probs],
            "Stacking Classifier": [f"{p:.2%}" for p in st_probs]
        })
        
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

        # =========================
        # TARJETAS FINALES COMPARATIVAS
        # =========================
        st.write("")
        st.subheader("Predicción Final de Cada Modelo")
        
        col1, col2, col3 = st.columns(3)
        
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

else:
    st.warning("Por favor, asegúrate de que los archivos 'RF.sav', 'XGBoost.sav' y 'Stacking.sav' estén en la misma carpeta.")