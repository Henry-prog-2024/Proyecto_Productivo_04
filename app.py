import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Compras - Inmobiliaria",
    page_icon="🏠",
    layout="wide"
)

# Título principal
st.title("🏠 Predictor de Probabilidad de Compra - Inmobiliaria")
st.markdown("---")

# Cargar el modelo y preprocesadores
@st.cache_resource
def load_model():
    try:
        model = joblib.load('mejor_modelo.pkl')
        scaler = joblib.load('scaler.pkl')
        columnas = joblib.load('columnas_modelo.pkl')

        # Cargar label encoders
        label_encoders = {}
        for col in ['proyecto', 'manzana', 'asesor', 'lote_ubicacion']:
            try:
                label_encoders[col] = joblib.load(f'label_encoder_{col}.pkl')
            except:
                st.warning(f"No se pudo cargar label_encoder_{col}.pkl")
                label_encoders[col] = None

        return model, scaler, columnas, label_encoders
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, None, None, None

model, scaler, columnas_modelo, label_encoders = load_model()
if model is None:
    st.stop()

# --- Pestañas principales ---
tab1, tab2 = st.tabs(["🔮 Predicción", "📜 Historial de Evaluaciones"])

with tab1:
    st.sidebar.header("📋 Datos del Cliente y Propiedad")

    # --- Secciones ---
    st.sidebar.subheader("1. Información del Proyecto")
    proyecto = st.sidebar.selectbox("Proyecto",
        [f"PROYECTO_{i}" for i in range(1, 11)])
    manzana = st.sidebar.selectbox("Manzana", ['Mz-A', 'Mz-B', 'Mz-C', 'Mz-D', 'Mz-E'])
    lote_ubicacion = st.sidebar.selectbox("Ubicación del Lote",
        [f"UBICACION_{i}" for i in range(1, 11)])

    st.sidebar.subheader("2. Características del Lote")
    metros_cuadrados = st.sidebar.slider("Metros Cuadrados", 80, 200, 140, 5)
    lote_precio_total = st.sidebar.selectbox("Precio Total del Lote ($)",
        list(range(15000, 41000, 1000)))

    st.sidebar.subheader("3. Información de Reserva")
    monto_reserva = st.sidebar.selectbox("Monto de Reserva ($)", [500, 600, 800, 900, 1000, 2000, 5000, 10000])
    tiempo_reserva_dias = st.sidebar.slider("Tiempo de Reserva (días)", 1, 90, 7, 1)
    dias_hasta_limite = st.sidebar.slider("Días hasta Fecha Límite", 1, 30, 15, 1)
    metodo_pago = st.sidebar.selectbox("Método de Pago", ['EFECTIVO', 'TARJETA', 'YAPE'])

    st.sidebar.subheader("4. Información del Cliente")
    dni_cliente = st.sidebar.text_input("DNI del Cliente (opcional)", "")
    cliente_edad = st.sidebar.slider("Edad del Cliente", 25, 70, 45, 1)
    cliente_genero = st.sidebar.selectbox("Género del Cliente", ['M', 'F'])
    cliente_profesion = st.sidebar.selectbox("Profesión del Cliente",
        ['Ingeniero', 'Doctor', 'Abogado', 'Docente', 'Comerciante', 'Empresario', 'Otro'])
    cliente_distrito = st.sidebar.selectbox("Distrito del Cliente",
        ['Distrito_A', 'Distrito_B', 'Distrito_C', 'Distrito_D', 'Distrito_E'])
    SALARIO_DECLARADO = st.sidebar.selectbox("Salario Aproximado ($)",
        [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])

    st.sidebar.subheader("5. Comportamiento y Características")
    n_visitas = st.sidebar.slider("Número de Visitas", 0, 5, 2, 1)
    canal_contacto = st.sidebar.selectbox("Canal de Contacto",
        ['EVENTO', 'FACEBOOK', 'PAGINA WEB', 'WHATSAPP', 'INSTAGRAM', 'VOLANTES'])
    asesor = st.sidebar.selectbox("Asesor", [f"Asesor_{i}" for i in range(1, 101)])
    promesa_regalo = st.sidebar.selectbox("Promesa de Regalo",
        ['Ninguno', 'Cocina', 'Refrigeradora', 'TV', 'Lavadora'])
    DOCUMENTOS = st.sidebar.selectbox("Estado de Documentos", ['Completo', 'Incompleto', 'Pendiente'])

    st.sidebar.subheader("6. Ubicación y Amenities")
    CERCA_AVENIDAS = st.sidebar.selectbox("Cerca de Avenidas", ['Si', 'No'])
    CERCA_COLEGIOS = st.sidebar.selectbox("Cerca de Colegios", ['Si', 'No'])
    CERCA_PARQUE = st.sidebar.selectbox("Cerca de Parques", ['Si', 'No'])

    # --- Función de preprocesamiento ---
    def preprocess_input(data):
        try:
            input_df = pd.DataFrame([data])
            input_df['ratio_reserva_precio'] = input_df['monto_reserva'] / input_df['lote_precio_total']
            input_df['precio_m2'] = input_df['lote_precio_total'] / input_df['metros_cuadrados']

            # Codificar edad
            input_df['cliente_edad_cat_36-45'] = int(36 <= data['cliente_edad'] <= 45)
            input_df['cliente_edad_cat_46-55'] = int(46 <= data['cliente_edad'] <= 55)
            input_df['cliente_edad_cat_56-70'] = int(56 <= data['cliente_edad'] <= 70)

            categorical_mappings = {
                'metodo_pago': ['EFECTIVO', 'TARJETA', 'YAPE'],
                'cliente_genero': ['M', 'F'],
                'cliente_profesion': ['Ingeniero', 'Doctor', 'Abogado', 'Docente', 'Comerciante', 'Empresario', 'Otro'],
                'cliente_distrito': ['Distrito_A', 'Distrito_B', 'Distrito_C', 'Distrito_D', 'Distrito_E'],
                'canal_contacto': ['EVENTO', 'FACEBOOK', 'PAGINA WEB', 'WHATSAPP', 'INSTAGRAM', 'VOLANTES'],
                'promesa_regalo': ['Ninguno', 'Cocina', 'Refrigeradora', 'TV', 'Lavadora'],
                'DOCUMENTOS': ['Completo', 'Incompleto', 'Pendiente'],
                'CERCA_AVENIDAS': ['Si', 'No'],
                'CERCA_COLEGIOS': ['Si', 'No'],
                'CERCA_PARQUE': ['Si', 'No']
            }
            for col, values in categorical_mappings.items():
                for value in values[1:]:
                    input_df[f"{col}_{value}"] = int(data[col] == value)

            # Label encoding
            for col in ['proyecto', 'manzana', 'asesor', 'lote_ubicacion']:
                if label_encoders.get(col):
                    try:
                        input_df[f"{col}_encoded"] = label_encoders[col].transform([data[col]])[0]
                    except:
                        input_df[f"{col}_encoded"] = 0

            # Asegurar columnas
            for col in columnas_modelo:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[columnas_modelo]

            numeric_cols = ['metros_cuadrados', 'monto_reserva', 'lote_precio_total',
                            'tiempo_reserva_dias', 'SALARIO_DECLARADO', 'n_visitas',
                            'ratio_reserva_precio', 'dias_hasta_limite', 'precio_m2']
            numeric_cols = [c for c in numeric_cols if c in input_df.columns]
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            return input_df
        except Exception as e:
            st.error(f"Error en preprocesamiento: {e}")
            return None

    # --- Botón de predicción ---
    if st.sidebar.button("🎯 Predecir Probabilidad de Compra", type="primary"):
        input_data = {
            'proyecto': proyecto, 'manzana': manzana, 'lote_ubicacion': lote_ubicacion,
            'metros_cuadrados': metros_cuadrados, 'lote_precio_total': lote_precio_total,
            'monto_reserva': monto_reserva, 'tiempo_reserva_dias': tiempo_reserva_dias,
            'dias_hasta_limite': dias_hasta_limite, 'metodo_pago': metodo_pago,
            'cliente_edad': cliente_edad, 'cliente_genero': cliente_genero,
            'cliente_profesion': cliente_profesion, 'cliente_distrito': cliente_distrito,
            'SALARIO_DECLARADO': SALARIO_DECLARADO, 'n_visitas': n_visitas,
            'canal_contacto': canal_contacto, 'asesor': asesor,
            'promesa_regalo': promesa_regalo, 'DOCUMENTOS': DOCUMENTOS,
            'CERCA_AVENIDAS': CERCA_AVENIDAS, 'CERCA_COLEGIOS': CERCA_COLEGIOS, 'CERCA_PARQUE': CERCA_PARQUE
        }

        processed_data = preprocess_input(input_data)
        if processed_data is not None:
            try:
                probabilidad = model.predict_proba(processed_data)[0][1]
                prediccion = model.predict(processed_data)[0]

                st.success("✅ Predicción completada!")

                # Resultados visuales
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Probabilidad de Compra", f"{probabilidad*100:.1f}%")
                    if probabilidad > 0.7:
                        st.success("🎉 Alta probabilidad de compra")
                    elif probabilidad > 0.4:
                        st.warning("⚠️ Probabilidad media de compra")
                    else:
                        st.error("📉 Baja probabilidad de compra")
                with col2:
                    st.progress(float(probabilidad))
                    st.caption(f"Confianza del modelo: {probabilidad*100:.1f}%")

                # --- Análisis detallado ---
                st.subheader("📊 Análisis de la Predicción")
                col3, col4 = st.columns(2)
                with col3:
                    st.info("**Factores Positivos:**")
                    if monto_reserva >= 2000: st.write("✅ Monto de reserva alto")
                    if n_visitas >= 3: st.write("✅ Múltiples visitas")
                    if DOCUMENTOS == 'Completo': st.write("✅ Documentación completa")
                    if SALARIO_DECLARADO >= 3000: st.write("✅ Buen nivel de ingresos")
                    if CERCA_AVENIDAS == 'Si': st.write("✅ Buena ubicación")

                with col4:
                    st.warning("**Factores de Riesgo:**")
                    if monto_reserva < 1000: st.write("❌ Monto de reserva bajo")
                    if n_visitas <= 2: st.write("❌ Pocas visitas")
                    if DOCUMENTOS == 'Incompleto': st.write("❌ Documentos incompletos")
                    if SALARIO_DECLARADO < 3000: st.write("❌ Ingreso bajo")

                # --- Guardar resultado ---
                historial_path = "historial_predicciones.csv"
                nueva_fila = pd.DataFrame([{
                    "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "DNI": dni_cliente or "N/A",
                    "Proyecto": proyecto,
                    "Probabilidad": round(probabilidad * 100, 2),
                    "Resultado": "Compra" if prediccion == 1 else "No Compra"
                }])
                if os.path.exists(historial_path):
                    historial_df = pd.read_csv(historial_path)
                    historial_df = pd.concat([historial_df, nueva_fila], ignore_index=True)
                else:
                    historial_df = nueva_fila
                historial_df.to_csv(historial_path, index=False)

                st.success("✅ Predicción guardada en historial.")
                with open(historial_path, "rb") as file:
                    st.download_button("📥 Descargar Historial", file, file_name="historial_predicciones.csv")

            except Exception as e:
                st.error(f"Error en la predicción: {e}")

with tab2:
    st.header("📜 Historial de Evaluaciones")
    if os.path.exists("historial_predicciones.csv"):
        df_hist = pd.read_csv("historial_predicciones.csv")
        st.dataframe(df_hist.sort_values("Fecha", ascending=False), use_container_width=True)
        with open("historial_predicciones.csv", "rb") as file:
            st.download_button("📥 Descargar Historial Completo", file, file_name="historial_predicciones.csv")
    else:
        st.info("Aún no hay predicciones registradas.")
