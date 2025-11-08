import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Compras - Inmobiliaria",
    page_icon="🏠",
    layout="wide"
)

# Título principal
st.title("🏠 Predictor de Compras - Inmobiliaria")

# --- Carga del modelo y el escalador ---
MODEL_PATH = "modelo_final_XGBoost.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ No se encontraron los archivos del modelo o del escalador.")
    st.stop()

modelo = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- Sección de entrada de datos ---
st.header("📋 Datos del Cliente")

# Campo de DNI (máximo 8 dígitos)
dni = st.text_input("DNI del Cliente (8 dígitos):", max_chars=8)
if dni and not dni.isdigit():
    st.warning("⚠️ El DNI debe contener solo números.")
elif dni and len(dni) != 8:
    st.warning("⚠️ El DNI debe tener exactamente 8 dígitos.")

# Ejemplo de campos adicionales (ajústalos según tu modelo)
edad = st.number_input("Edad del cliente", min_value=18, max_value=100, value=30)
salario = st.number_input("Salario aproximado (S/)", min_value=0.0, step=100.0, value=2000.0)
dias_limite = st.number_input("Días hasta la fecha límite", min_value=0, max_value=365, value=30)

# --- Botón para hacer la predicción ---
if st.button("🔍 Analizar Probabilidad de Compra"):

    # Validar DNI antes de continuar
    if not dni or len(dni) != 8 or not dni.isdigit():
        st.error("❌ Ingresa un DNI válido de 8 dígitos antes de continuar.")
    else:
        # Crear dataframe con los datos
        datos_cliente = pd.DataFrame({
            "Edad": [edad],
            "Salario": [salario],
            "Dias_Limite": [dias_limite]
        })

        # Escalar los datos
        datos_escalados = scaler.transform(datos_cliente)

        # Hacer la predicción
        prediccion = modelo.predict(datos_escalados)[0]
        probabilidad = modelo.predict_proba(datos_escalados)[0][1]

        # Mostrar resultados
        st.subheader("📊 Resultado de la Evaluación")
        if prediccion == 1:
            st.success(f"✅ Alta probabilidad de compra ({probabilidad:.2%})")
        else:
            st.warning(f"⚠️ Baja probabilidad de compra ({probabilidad:.2%})")

        # --- Botón para guardar la evaluación ---
        if st.button("💾 Guardar Evaluación"):
            historial_path = "historial_predicciones.csv"

            nueva_fila = pd.DataFrame({
                "Fecha": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "DNI": [dni],
                "Edad": [edad],
                "Salario": [salario],
                "Dias_Limite": [dias_limite],
                "Probabilidad_Compra": [round(probabilidad, 4)],
                "Resultado": ["Alta" if prediccion == 1 else "Baja"]
            })

            if os.path.exists(historial_path):
                historial = pd.read_csv(historial_path)
                historial = pd.concat([historial, nueva_fila], ignore_index=True)
            else:
                historial = nueva_fila

            historial.to_csv(historial_path, index=False)
            st.success("💾 Evaluación guardada correctamente.")

# --- Mostrar historial acumulado ---
st.header("📚 Historial de Predicciones")

historial_path = "historial_predicciones.csv"
if os.path.exists(historial_path):
    historial = pd.read_csv(historial_path)
    st.dataframe(historial, use_container_width=True)
    st.download_button(
        label="📥 Descargar Historial",
        data=historial.to_csv(index=False).encode("utf-8"),
        file_name="historial_predicciones.csv",
        mime="text/csv"
    )
else:
    st.info("No hay predicciones guardadas aún.")
