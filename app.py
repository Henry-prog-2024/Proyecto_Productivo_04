import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io

# ==========================
# ⚙️ CONFIGURACIÓN DE PÁGINA
# ==========================
st.set_page_config(
    page_title="Predictor de Compras - Inmobiliaria",
    page_icon="🏠",
    layout="wide"
)

# ==========================
# 📦 CARGAR MODELO Y SCALER
# ==========================
@st.cache_resource
def load_model():
    # Cargar modelo y preprocesadores (sin "_balanceado")
    model = joblib.load('mejor_modelo.pkl')
    scaler = joblib.load('scaler.pkl')
    columnas = joblib.load('columnas_modelo.pkl')
    return model, scaler

model, scaler = load_model()

# ==========================
# 🧮 FUNCIÓN DE PREPROCESAMIENTO
# ==========================
def preprocess_input(data):
    try:
        input_df = pd.DataFrame([data])

        # Asegurar existencia de 'dias_hasta_limite'
        if 'dias_hasta_limite' not in input_df.columns:
            input_df['dias_hasta_limite'] = 30  # valor por defecto

        numeric_cols = ['metros_cuadrados', 'lote_precio_total', 'monto_reserva',
                        'tiempo_reserva_dias', 'SALARIO_DECLARADO', 'n_visitas', 'dias_hasta_limite']

        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Codificación de edad categorizada
        edad = input_df['cliente_edad'].iloc[0]
        input_df['cliente_edad_cat_36-45'] = 1 if 36 <= edad <= 45 else 0
        input_df['cliente_edad_cat_46-55'] = 1 if 46 <= edad <= 55 else 0
        input_df['cliente_edad_cat_56-70'] = 1 if 56 <= edad <= 70 else 0

        return input_df
    except Exception as e:
        st.error(f"Error en preprocesamiento: {e}")
        return None

# ==========================
# 🧭 NAVEGACIÓN ENTRE PÁGINAS
# ==========================
tabs = st.tabs(["🔮 Predicción", "📋 Historial de Evaluaciones"])

# ===================================================
# 🧩 PANTALLA 1: PREDICCIÓN DE PROBABILIDAD DE COMPRA
# ===================================================
with tabs[0]:

    st.title("🏠 Predictor de Probabilidad de Compra")

    st.sidebar.header("📥 Ingreso de Datos del Cliente")

    # Datos básicos
    dni_cliente = st.sidebar.text_input("DNI del Cliente", "")
    proyecto = st.sidebar.selectbox("Proyecto", ["Residencial Sol", "Jardines del Valle", "Mirador Real"])
    manzana = st.sidebar.text_input("Manzana", "A")
    lote_ubicacion = st.sidebar.text_input("Ubicación del lote", "Lote 01")

    metros_cuadrados = st.sidebar.number_input("Metros cuadrados", 20, 500, 100)
    lote_precio_total = st.sidebar.number_input("Precio total del lote (S/)", 10000, 200000, 50000)
    monto_reserva = st.sidebar.number_input("Monto de reserva (S/)", 100, 5000, 1000)
    tiempo_reserva_dias = st.sidebar.slider("Tiempo de reserva (días)", 0, 90, 30)
    dias_hasta_limite = st.sidebar.slider("Días hasta la fecha límite", 0, 365, 30)

    metodo_pago = st.sidebar.selectbox("Método de pago", ["Crédito", "Contado"])
    cliente_edad = st.sidebar.number_input("Edad del cliente", 18, 80, 35)
    cliente_genero = st.sidebar.selectbox("Género", ["Masculino", "Femenino"])
    cliente_profesion = st.sidebar.text_input("Profesión", "Empleado")
    cliente_distrito = st.sidebar.text_input("Distrito", "Huancayo")

    SALARIO_DECLARADO = st.sidebar.number_input("Salario declarado (S/)", 1000, 20000, 3000)
    n_visitas = st.sidebar.slider("Número de visitas", 1, 10, 3)
    canal_contacto = st.sidebar.selectbox("Canal de contacto", ["Redes", "Presencial", "Referido"])
    asesor = st.sidebar.text_input("Asesor", "María López")
    promesa_regalo = st.sidebar.selectbox("Promesa de regalo", ["Sí", "No"])
    DOCUMENTOS = st.sidebar.selectbox("Documentación", ["Completo", "Incompleto"])
    CERCA_AVENIDAS = st.sidebar.selectbox("Cerca de avenidas", ["Sí", "No"])
    CERCA_COLEGIOS = st.sidebar.selectbox("Cerca de colegios", ["Sí", "No"])
    CERCA_PARQUE = st.sidebar.selectbox("Cerca de parque", ["Sí", "No"])

    st.sidebar.markdown("---")

    if st.sidebar.button("🎯 Predecir Probabilidad de Compra", type="primary"):
        input_data = {
            'proyecto': proyecto,
            'manzana': manzana,
            'lote_ubicacion': lote_ubicacion,
            'metros_cuadrados': metros_cuadrados,
            'lote_precio_total': lote_precio_total,
            'monto_reserva': monto_reserva,
            'tiempo_reserva_dias': tiempo_reserva_dias,
            'dias_hasta_limite': dias_hasta_limite,
            'metodo_pago': metodo_pago,
            'cliente_edad': cliente_edad,
            'cliente_genero': cliente_genero,
            'cliente_profesion': cliente_profesion,
            'cliente_distrito': cliente_distrito,
            'SALARIO_DECLARADO': SALARIO_DECLARADO,
            'n_visitas': n_visitas,
            'canal_contacto': canal_contacto,
            'asesor': asesor,
            'promesa_regalo': promesa_regalo,
            'DOCUMENTOS': DOCUMENTOS,
            'CERCA_AVENIDAS': CERCA_AVENIDAS,
            'CERCA_COLEGIOS': CERCA_COLEGIOS,
            'CERCA_PARQUE': CERCA_PARQUE,
            'DNI': dni_cliente
        }

        processed_data = preprocess_input(input_data)

        if processed_data is not None:
            try:
                probabilidad = model.predict_proba(processed_data)[0][1]
                prediccion = model.predict(processed_data)[0]

                st.success(f"✅ Predicción completada para el cliente DNI **{dni_cliente}**")

                # Mostrar resultado
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Probabilidad de Compra", f"{probabilidad*100:.1f}%")
                with col2:
                    st.progress(float(probabilidad))

                # Mensaje interpretativo
                if probabilidad > 0.7:
                    st.success("🎉 Alta probabilidad de compra")
                elif probabilidad > 0.4:
                    st.warning("⚠️ Probabilidad media de compra")
                else:
                    st.error("📉 Baja probabilidad de compra")

                # ================================
                # 📊 Análisis de Factores
                # ================================
                st.subheader("📊 Análisis de la Predicción")

                col3, col4 = st.columns(2)

                with col3:
                    st.info("**Factores Positivos:**")
                    if monto_reserva >= 2000:
                        st.write("✅ Monto de reserva alto")
                    if n_visitas >= 3:
                        st.write("✅ Múltiples visitas")
                    if DOCUMENTOS == 'Completo':
                        st.write("✅ Documentación completa")
                    if SALARIO_DECLARADO >= 3000:
                        st.write("✅ Buen nivel de ingresos")

                with col4:
                    st.warning("**Factores de Riesgo:**")
                    if monto_reserva < 1000:
                        st.write("❌ Monto de reserva bajo")
                    if n_visitas <= 2:
                        st.write("❌ Pocas visitas")
                    if DOCUMENTOS == 'Incompleto':
                        st.write("❌ Documentación incompleta")
                    if SALARIO_DECLARADO < 3000:
                        st.write("❌ Bajo nivel de ingresos")

                # ========================================
                # 💾 BOTÓN GUARDAR PREDICCIÓN
                # ========================================
                registro_completo = input_data.copy()
                registro_completo["Probabilidad_Compra"] = probabilidad
                registro_completo["Predicción"] = "COMPRA" if prediccion == 1 else "NO COMPRA"
                df_registro = pd.DataFrame([registro_completo])
                archivo_csv = "evaluaciones_clientes.csv"

                if st.button("💾 Guardar esta predicción", type="primary"):
                    if os.path.exists(archivo_csv):
                        df_registro.to_csv(archivo_csv, mode='a', header=False, index=False)
                    else:
                        df_registro.to_csv(archivo_csv, index=False)
                    st.success("✅ Predicción guardada correctamente en 'evaluaciones_clientes.csv'")

            except Exception as e:
                st.error(f"Error en la predicción: {e}")

# ===================================================
# 🧩 PANTALLA 2: HISTORIAL DE EVALUACIONES
# ===================================================
with tabs[1]:
    st.title("📋 Historial de Evaluaciones")

    archivo_csv = "evaluaciones_clientes.csv"
    if os.path.exists(archivo_csv):
        df_historial = pd.read_csv(archivo_csv)
        st.dataframe(df_historial)

        csv_buffer = io.StringIO()
        df_historial.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Descargar historial completo",
            data=csv_buffer.getvalue(),
            file_name="evaluaciones_clientes.csv",
            mime="text/csv",
            key="btn_descargar_historial"
        )
    else:
        st.info("No hay evaluaciones registradas aún.")
