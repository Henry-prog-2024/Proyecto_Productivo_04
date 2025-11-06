import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
from io import StringIO

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

        label_encoders = {}
        for col in ['proyecto', 'manzana', 'asesor', 'lote_ubicacion']:
            try:
                label_encoders[col] = joblib.load(f'label_encoder_{col}.pkl')
            except:
                label_encoders[col] = None

        return model, scaler, columnas, label_encoders
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, None, None, None

# Cargar recursos
model, scaler, columnas_modelo, label_encoders = load_model()
if model is None:
    st.stop()

# Sidebar para entrada de datos
st.sidebar.header("📋 Datos del Cliente y Propiedad")

# DNI
dni_cliente = st.sidebar.text_input("🪪 DNI del Cliente", "")

# Información del Proyecto
st.sidebar.subheader("1. Información del Proyecto")
proyecto = st.sidebar.selectbox("Proyecto",
    [f'PROYECTO_{i}' for i in range(1, 11)]
)
manzana = st.sidebar.selectbox("Manzana", ['Mz-A', 'Mz-B', 'Mz-C', 'Mz-D', 'Mz-E'])
lote_ubicacion = st.sidebar.selectbox("Ubicación del Lote",
    [f'UBICACION_{i}' for i in range(1, 11)]
)

# Características del Lote
st.sidebar.subheader("2. Características del Lote")
metros_cuadrados = st.sidebar.slider("Metros Cuadrados", 80, 200, 140, 5)
lote_precio_total = st.sidebar.selectbox("Precio Total del Lote ($)", list(range(15000, 41000, 1000)))

# Información de Reserva
st.sidebar.subheader("3. Información de Reserva")
monto_reserva = st.sidebar.selectbox("Monto de Reserva ($)", [500, 600, 800, 900, 1000, 2000, 5000, 10000])
tiempo_reserva_dias = st.sidebar.slider("Tiempo de Reserva (días)", 1, 90, 7, 1)
dias_hasta_limite = 30  # Valor fijo para compatibilidad
metodo_pago = st.sidebar.selectbox("Método de Pago", ['EFECTIVO', 'TARJETA', 'YAPE'])

# Información del Cliente
st.sidebar.subheader("4. Información del Cliente")
cliente_edad = st.sidebar.slider("Edad del Cliente", 25, 70, 45, 1)
cliente_genero = st.sidebar.selectbox("Género del Cliente", ['M', 'F'])
cliente_profesion = st.sidebar.selectbox("Profesión del Cliente",
    ['Ingeniero', 'Doctor', 'Abogado', 'Docente', 'Comerciante', 'Empresario', 'Otro']
)
cliente_distrito = st.sidebar.selectbox("Distrito del Cliente",
    ['Distrito_A', 'Distrito_B', 'Distrito_C', 'Distrito_D', 'Distrito_E']
)
SALARIO_DECLARADO = st.sidebar.selectbox("Salario Aproximado ($)", [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])

# Comportamiento
st.sidebar.subheader("5. Comportamiento y Características")
n_visitas = st.sidebar.slider("Número de Visitas", 0, 5, 2, 1)
canal_contacto = st.sidebar.selectbox("Canal de Contacto",
    ['EVENTO', 'FACEBOOK', 'PAGINA WEB', 'WHATSAPP', 'INSTAGRAM', 'VOLANTES']
)
asesor = st.sidebar.selectbox("Asesor", [f"Asesor_{i}" for i in range(1, 101)])
promesa_regalo = st.sidebar.selectbox("Promesa de Regalo", ['Ninguno', 'Cocina', 'Refrigeradora', 'TV', 'Lavadora'])
DOCUMENTOS = st.sidebar.selectbox("Estado de Documentos", ['Completo', 'Incompleto', 'Pendiente'])

# Ubicación
st.sidebar.subheader("6. Ubicación y Amenities")
CERCA_AVENIDAS = st.sidebar.selectbox("Cerca de Avenidas", ['Si', 'No'])
CERCA_COLEGIOS = st.sidebar.selectbox("Cerca de Colegios", ['Si', 'No'])
CERCA_PARQUE = st.sidebar.selectbox("Cerca de Parques", ['Si', 'No'])

# Preprocesamiento
def preprocess_input(data):
    try:
        input_df = pd.DataFrame([data])
        input_df['ratio_reserva_precio'] = input_df['monto_reserva'] / input_df['lote_precio_total']
        input_df['precio_m2'] = input_df['lote_precio_total'] / input_df['metros_cuadrados']

        # Codificar edad categorizada
        if input_df['cliente_edad'].iloc[0] <= 35:
            input_df['cliente_edad_cat_36-45'] = 0
            input_df['cliente_edad_cat_46-55'] = 0
            input_df['cliente_edad_cat_56-70'] = 0
        elif input_df['cliente_edad'].iloc[0] <= 45:
            input_df['cliente_edad_cat_36-45'] = 1
            input_df['cliente_edad_cat_46-55'] = 0
            input_df['cliente_edad_cat_56-70'] = 0
        elif input_df['cliente_edad'].iloc[0] <= 55:
            input_df['cliente_edad_cat_36-45'] = 0
            input_df['cliente_edad_cat_46-55'] = 1
            input_df['cliente_edad_cat_56-70'] = 0
        else:
            input_df['cliente_edad_cat_36-45'] = 0
            input_df['cliente_edad_cat_46-55'] = 0
            input_df['cliente_edad_cat_56-70'] = 1

        # One-Hot Encoding
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
                col_name = f"{col}_{value}"
                input_df[col_name] = 1 if data[col] == value else 0

        for col in ['proyecto', 'manzana', 'asesor', 'lote_ubicacion']:
            if label_encoders.get(col) is not None:
                input_df[f'{col}_encoded'] = label_encoders[col].transform([data[col]])[0]
            else:
                input_df[f'{col}_encoded'] = 0

        for col in columnas_modelo:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[columnas_modelo]

        # Escalado seguro
        try:
            valid_cols = [c for c in scaler.feature_names_in_ if c in input_df.columns]
            input_df[valid_cols] = scaler.transform(input_df[valid_cols])
        except Exception as e:
            st.warning(f"No se pudieron escalar todas las variables numéricas: {e}")

        return input_df
    except Exception as e:
        st.error(f"Error en preprocesamiento: {e}")
        return None

# Inicializar DataFrame de resultados
if "evaluaciones" not in st.session_state:
    st.session_state.evaluaciones = pd.DataFrame()

# Botón de predicción
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
        'CERCA_PARQUE': CERCA_PARQUE
    }

    processed_data = preprocess_input(input_data)

    if processed_data is not None:
        prob = model.predict_proba(processed_data)[0][1]
        pred = model.predict(processed_data)[0]
        st.success(f"✅ Predicción completada para el cliente DNI **{dni_cliente}**")
        st.metric("Probabilidad de Compra", f"{prob*100:.1f}%")
        st.progress(float(prob))

        # Guardar resultado
        nueva_eval = pd.DataFrame([{
            "DNI": dni_cliente,
            "Probabilidad_Compra": prob,
            "Predicción": "COMPRA" if pred == 1 else "NO COMPRA"
        }])
        st.session_state.evaluaciones = pd.concat([st.session_state.evaluaciones, nueva_eval], ignore_index=True)

# Botón para descargar evaluaciones
if not st.session_state.evaluaciones.empty:
    csv = st.session_state.evaluaciones.to_csv(index=False)
    st.download_button(
        label="💾 Descargar Evaluaciones",
        data=csv,
        file_name="evaluaciones_clientes.csv",
        mime="text/csv"
    )