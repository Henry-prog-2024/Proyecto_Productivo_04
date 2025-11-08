import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Compras - Inmobiliaria",
    page_icon="🏠",
    layout="wide"
)

# Título principal
st.title("🏠 Predictor de Probabilidad de Compra - Inmobiliaria")

# Cargar modelo y scaler
modelo_path = "modelo_final_XGBoost.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(modelo_path) or not os.path.exists(scaler_path):
    st.error("❌ No se encontró el archivo del modelo o del scaler.")
    st.stop()

modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)

# Inicializar historial en sesión
if "historial" not in st.session_state:
    st.session_state.historial = []

# Ingreso de DNI
dni_cliente = st.text_input("🆔 Ingrese el DNI del cliente:")

# Campos de entrada
st.subheader("📋 Información del Cliente")
col1, col2, col3, col4 = st.columns(4)
with col1:
    ingreso = st.number_input("💰 Ingreso mensual (S/)", min_value=0.0, value=2500.0)
with col2:
    edad = st.number_input("🎂 Edad", min_value=18, value=35)
with col3:
    antiguedad = st.number_input("📆 Antigüedad laboral (años)", min_value=0, value=3)
with col4:
    cuota_inicial = st.number_input("🏦 Cuota inicial (%)", min_value=0, max_value=100, value=20)

# Botón de predicción
if st.button("📊 Predecir Probabilidad"):
    try:
        datos = np.array([[ingreso, edad, antiguedad, cuota_inicial]])
        datos_escalados = scaler.transform(datos)
        probabilidad = modelo.predict_proba(datos_escalados)[0][1]
        prediccion = modelo.predict(datos_escalados)[0]

        st.markdown("---")
        st.subheader("📈 Resultado de la Predicción")
        st.metric("Probabilidad de Compra (%)", f"{probabilidad*100:.2f}")
        st.write("🔍 Interpretación:", "Alta probabilidad de compra 🟢" if prediccion == 1 else "Baja probabilidad de compra 🔴")

        # Botón para guardar la evaluación
        if st.button("💾 Guardar Evaluación"):
            if dni_cliente.strip() == "":
                st.warning("⚠️ Ingresa un DNI antes de guardar la evaluación.")
            else:
                nuevo_registro = {
                    "DNI": dni_cliente,
                    "Ingreso": ingreso,
                    "Edad": edad,
                    "Antigüedad (años)": antiguedad,
                    "Cuota Inicial (%)": cuota_inicial,
                    "Probabilidad (%)": round(probabilidad * 100, 2),
                    "Resultado": "Compra" if prediccion == 1 else "No Compra"
                }
                st.session_state.historial.append(nuevo_registro)
                st.success("✅ Evaluación guardada correctamente.")

                # Mostrar cuadro de historial debajo del botón
                if len(st.session_state.historial) > 0:
                    st.markdown("### 📜 Evaluaciones Realizadas Hasta el Momento")
                    df_historial = pd.DataFrame(st.session_state.historial)
                    st.dataframe(df_historial, use_container_width=True)

                    # Botón para descargar CSV
                    csv = df_historial.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Descargar Historial en CSV",
                        data=csv,
                        file_name="historial_predicciones.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"⚠️ Error al realizar la predicción: {e}")

# Mostrar historial general si existen registros
if len(st.session_state.historial)_
