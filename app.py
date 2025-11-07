import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
import io

# ==========================================================
# 🏠 CONFIGURACIÓN DE LA PÁGINA
# ==========================================================
st.set_page_config(
    page_title="Predictor de Compras - Inmobiliaria",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Predictor de Probabilidad de Compra - Inmobiliaria")
st.markdown("---")

# ==========================================================
# 📦 CARGA DE MODELOS Y PREPROCESADORES
# ==========================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('mejor_modelo.pkl')
        scaler = joblib.load('scaler.pkl')
        columnas = joblib.load('columnas_modelo.pkl')

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

# ==========================================================
# 🧭 PESTAÑAS PRINCIPALES
# ==========================================================
tab_prediccion, tab_historial = st.tabs(["🎯 Predicción de Compra", "📜 Historial de Evaluaciones"])

# ==========================================================
# 🧮 TAB 1: PREDICCIÓN DE COMPRA
# ==========================================================
with tab_prediccion:

    # --- Sidebar para datos del cliente ---
    st.sidebar.header("📋 Datos del Cliente y Propiedad")
    dni_cliente = st.sidebar.text_input("🪪 DNI del Cliente", "")

    st.sidebar.subheader("1. Información del Proyecto")
    proyecto = st.sidebar.selectbox("Proyecto", [f"PROYECTO_{i}" for i in range(1, 11)])
    manzana = st.sidebar.selectbox("Manzana", ['Mz-A', 'Mz-B', 'Mz-C', 'Mz-D', 'Mz-E'])
    lote_ubicacion = st.sidebar.selectbox("Ubicación del Lote", [f"UBICACION_{i}" for i in range(1, 11)])

    st.sidebar.subheader("2. Características del Lote")
    metros_cuadrados = st.sidebar.slider("Metros Cuadrados", 80, 200, 140, 5)
    lote_precio_total = st.sidebar.selectbox("Precio Total del Lote ($)", list(range(15000, 41000, 1000)))

    st.sidebar.subheader("3. Información de Reserva")
    monto_reserva = st.sidebar.selectbox("Monto de Reserva ($)", [500, 600, 800, 900, 1000, 2000, 5000, 10000])
    tiempo_reserva_dias = st.sidebar.slider("Tiempo de Reserva (días)", 1, 90, 7, 1)
    metodo_pago = st.sidebar.selectbox("Método de Pago", ['EFECTIVO', 'TARJETA', 'YAPE'])

    st.sidebar.subheader("4. Información del Cliente")
    cliente_edad = st.sidebar.slider("Edad del Cliente", 25, 70, 45, 1)
    cliente_genero = st.sidebar.selectbox("Género del Cliente", ['M', 'F'])
    cliente_profesion = st.sidebar.selectbox(
        "Profesión del Cliente",
        ['Ingeniero', 'Doctor', 'Abogado', 'Docente', 'Comerciante', 'Empresario', 'Otro']
    )
    cliente_distrito = st.sidebar.selectbox(
        "Distrito del Cliente",
        ['Distrito_A', 'Distrito_B', 'Distrito_C', 'Distrito_D', 'Distrito_E']
    )
    SALARIO_DECLARADO = st.sidebar.selectbox(
        "Salario Aproximado ($)",
        [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    )

    st.sidebar.subheader("5. Comportamiento y Características")
    n_visitas = st.sidebar.slider("Número de Visitas", 0, 5, 2, 1)
    canal_contacto = st.sidebar.selectbox(
        "Canal de Contacto",
        ['EVENTO', 'FACEBOOK', 'PAGINA WEB', 'WHATSAPP', 'INSTAGRAM', 'VOLANTES']
    )
    asesor = st.sidebar.selectbox("Asesor", [f"Asesor_{i}" for i in range(1, 101)])
    promesa_regalo = st.sidebar.selectbox(
        "Promesa de Regalo", ['Ninguno', 'Cocina', 'Refrigeradora', 'TV', 'Lavadora']
    )
    DOCUMENTOS = st.sidebar.selectbox("Estado de Documentos", ['Completo', 'Incompleto', 'Pendiente'])

    st.sidebar.subheader("6. Ubicación y Amenities")
    CERCA_AVENIDAS = st.sidebar.selectbox("Cerca de Avenidas", ['Si', 'No'])
    CERCA_COLEGIOS = st.sidebar.selectbox("Cerca de Colegios", ['Si', 'No'])
    CERCA_PARQUE = st.sidebar.selectbox("Cerca de Parques", ['Si', 'No'])

    # --- Preprocesamiento ---
    def preprocess_input(data):
        try:
            input_df = pd.DataFrame([data])
            input_df['ratio_reserva_precio'] = input_df['monto_reserva'] / input_df['lote_precio_total']
            input_df['precio_m2'] = input_df['lote_precio_total'] / input_df['metros_cuadrados']

            # Edad categórica
            if data['cliente_edad'] <= 35:
                input_df['cliente_edad_cat_36-45'] = 0
                input_df['cliente_edad_cat_46-55'] = 0
                input_df['cliente_edad_cat_56-70'] = 0
            elif data['cliente_edad'] <= 45:
                input_df['cliente_edad_cat_36-45'] = 1
                input_df['cliente_edad_cat_46-55'] = 0
                input_df['cliente_edad_cat_56-70'] = 0
            elif data['cliente_edad'] <= 55:
                input_df['cliente_edad_cat_36-45'] = 0
                input_df['cliente_edad_cat_46-55'] = 1
                input_df['cliente_edad_cat_56-70'] = 0
            else:
                input_df['cliente_edad_cat_36-45'] = 0
                input_df['cliente_edad_cat_46-55'] = 0
                input_df['cliente_edad_cat_56-70'] = 1

            # One-Hot encoding
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
                    input_df[f"{col}_{value}"] = 1 if data[col] == value else 0

            # Label encoding
            for col in ['proyecto', 'manzana', 'asesor', 'lote_ubicacion']:
                if label_encoders.get(col) is not None:
                    try:
                        input_df[f"{col}_encoded"] = label_encoders[col].transform([data[col]])[0]
                    except:
                        input_df[f"{col}_encoded"] = 0

            # Asegurar columnas
            for col in columnas_modelo:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[columnas_modelo]

            numeric_cols = [
                'metros_cuadrados', 'monto_reserva', 'lote_precio_total',
                'tiempo_reserva_dias', 'SALARIO_DECLARADO', 'n_visitas',
                'ratio_reserva_precio', 'precio_m2'
            ]
            numeric_cols = [c for c in numeric_cols if c in input_df.columns]
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

            return input_df

        except Exception as e:
            st.error(f"Error en preprocesamiento: {e}")
            return None

    # --- BOTÓN DE PREDICCIÓN ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🎯 Predecir Probabilidad de Compra", type="primary"):
        dias_hasta_limite = 30  # valor promedio o neutro
        input_data = {
            'proyecto': proyecto,
            'manzana': manzana,
            'lote_ubicacion': lote_ubicacion,
            'metros_cuadrados': metros_cuadrados,
            'lote_precio_total': lote_precio_total,
            'monto_reserva': monto_reserva,
            'tiempo_reserva_dias': tiempo_reserva_dias,
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
            'DNI': dni_cliente,
            'dias_hasta_limite': dias_hasta_limite
        }

        processed_data = preprocess_input(input_data)
        if processed_data is not None:
            probabilidad = model.predict_proba(processed_data)[0][1]
            prediccion = model.predict(processed_data)[0]

            st.metric("Probabilidad de Compra", f"{probabilidad*100:.1f}%")
            st.progress(float(probabilidad))

            if probabilidad > 0.7:
                st.success("🎉 Alta probabilidad de compra")
            elif probabilidad > 0.4:
                st.warning("⚠️ Probabilidad media de compra")
            else:
                st.error("📉 Baja probabilidad de compra")

            registro_completo = input_data.copy()
            registro_completo["Probabilidad_Compra"] = probabilidad
            registro_completo["Predicción"] = "COMPRA" if prediccion == 1 else "NO COMPRA"

            if st.button("💾 Guardar esta predicción en el historial", key="guardar_prediccion"):
                archivo_csv = "evaluaciones_clientes.csv"
                df_registro = pd.DataFrame([registro_completo])

                if os.path.exists(archivo_csv):
                    df_historial = pd.read_csv(archivo_csv, on_bad_lines='skip')
                    df_historial = pd.concat([df_historial, df_registro], ignore_index=True)
                else:
                    df_historial = df_registro

                df_historial.to_csv(archivo_csv, index=False)
                st.success("✅ Predicción guardada correctamente.")
                st.session_state["historial"] = df_historial

# ==========================================================
# 🧾 TAB 2: HISTORIAL DE EVALUACIONES
# ==========================================================
with tab_historial:
    st.subheader("📜 Historial de Evaluaciones Guardadas")
    archivo_csv = "evaluaciones_clientes.csv"

    if os.path.exists(archivo_csv):
        df_historial = pd.read_csv(archivo_csv, on_bad_lines='skip')
        st.dataframe(df_historial)

        csv_buffer = io.StringIO()
        df_historial.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Descargar historial completo",
            data=csv_buffer.getvalue(),
            file_name="evaluaciones_clientes.csv",
            mime="text/csv"
        )
    else:
        st.info("Aún no hay evaluaciones guardadas.")
