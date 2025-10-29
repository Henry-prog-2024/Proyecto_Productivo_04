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
st.markdown("---")

# Cargar el modelo y preprocesadores
@st.cache_resource
def load_model():
    try:
        # En Colab, los archivos están en el directorio actual
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
        st.info("Asegúrate de tener todos los archivos .pkl en el directorio actual")
        return None, None, None, None

# Cargar recursos
model, scaler, columnas_modelo, label_encoders = load_model()

if model is None:
    st.stop()

# Sidebar para entrada de datos
st.sidebar.header("📋 Datos del Cliente y Propiedad")

# Dividir en secciones
st.sidebar.subheader("1. Información del Proyecto")

proyecto = st.sidebar.selectbox(
    "Proyecto",
    ['PROYECTO_1', 'PROYECTO_2', 'PROYECTO_3', 'PROYECTO_4', 'PROYECTO_5',
     'PROYECTO_6', 'PROYECTO_7', 'PROYECTO_8', 'PROYECTO_9', 'PROYECTO_10']
)

manzana = st.sidebar.selectbox(
    "Manzana",
    ['Mz-A', 'Mz-B', 'Mz-C', 'Mz-D', 'Mz-E']
)

lote_ubicacion = st.sidebar.selectbox(
    "Ubicación del Lote",
    ['UBICACION_1', 'UBICACION_2', 'UBICACION_3', 'UBICACION_4', 'UBICACION_5',
     'UBICACION_6', 'UBICACION_7', 'UBICACION_8', 'UBICACION_9', 'UBICACION_10']
)

st.sidebar.subheader("2. Características del Lote")

metros_cuadrados = st.sidebar.slider(
    "Metros Cuadrados",
    min_value=80,
    max_value=200,
    value=140,
    step=5
)

lote_precio_total = st.sidebar.selectbox(
    "Precio Total del Lote ($)",
    [15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000,
     25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000, 34000,
     35000, 36000, 37000, 38000, 39000, 40000]
)

st.sidebar.subheader("3. Información de Reserva")

monto_reserva = st.sidebar.selectbox(
    "Monto de Reserva ($)",
    [500, 600, 800, 900, 1000,2000,5000,10000]
)

tiempo_reserva_dias = st.sidebar.slider(
    "Tiempo de Reserva (días)",
    min_value=1,
    max_value=730,
    value=180,
    step=1
)

#dias_hasta_limite = st.sidebar.slider(
 #   "Días hasta Fecha Límite",
  #  min_value=1,
   # max_value=30,
    #value=30,
    #step=1
#)

metodo_pago = st.sidebar.selectbox(
    "Método de Pago",
    ['EFECTIVO', 'TARJETA', 'YAPE']
)

st.sidebar.subheader("4. Información del Cliente")

cliente_edad = st.sidebar.slider(
    "Edad del Cliente",
    min_value=25,
    max_value=70,
    value=45,
    step=1
)

cliente_genero = st.sidebar.selectbox(
    "Género del Cliente",
    ['M', 'F']
)

cliente_profesion = st.sidebar.selectbox(
    "Profesión del Cliente",
    ['Ingeniero', 'Doctor', 'Abogado', 'Docente', 'Comerciante', 'Empresario', 'Otro']
)

cliente_distrito = st.sidebar.selectbox(
    "Distrito del Cliente",
    ['Distrito_A', 'Distrito_B', 'Distrito_C', 'Distrito_D', 'Distrito_E']
)

SALARIO_APROX = st.sidebar.selectbox(
    "Salario Aproximado ($)",
    [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
)

st.sidebar.subheader("5. Comportamiento y Características")

n_visitas = st.sidebar.slider(
    "Número de Visitas",
    min_value=0,
    max_value=5,
    value=2,
    step=1
)

canal_contacto = st.sidebar.selectbox(
    "Canal de Contacto",
    ['EVENTO', 'FACEBOOK', 'PAGINA WEB', 'WHATSAPP', 'INSTAGRAM', 'VOLANTES']
)

asesor = st.sidebar.selectbox(
    "Asesor",
    [f"Asesor_{i}" for i in range(1, 101)]
)

promesa_regalo = st.sidebar.selectbox(
    "Promesa de Regalo",
    ['Ninguno', 'Cocina', 'Refrigeradora', 'TV', 'Lavadora']
)

DOCUMENTOS = st.sidebar.selectbox(
    "Estado de Documentos",
    ['Completo', 'Incompleto', 'Pendiente']
)

st.sidebar.subheader("6. Ubicación y Amenities")

CERCA_AVENIDAS = st.sidebar.selectbox(
    "Cerca de Avenidas",
    ['Si', 'No']
)

CERCA_COLEGIOS = st.sidebar.selectbox(
    "Cerca de Colegios",
    ['Si', 'No']
)

CERCA_PARQUE = st.sidebar.selectbox(
    "Cerca de Parques",
    ['Si', 'No']
)

# Función para preprocesar los datos de entrada
def preprocess_input(data):
    try:
        # Crear DataFrame
        input_df = pd.DataFrame([data])

        # Feature Engineering
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

        # One-Hot Encoding manual para otras variables
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
            for value in values[1:]:  # Skip first category (drop_first=True)
                col_name = f"{col}_{value}"
                input_df[col_name] = 1 if data[col] == value else 0

        # Label Encoding para variables con muchos valores
        for col in ['proyecto', 'manzana', 'asesor', 'lote_ubicacion']:
            if label_encoders.get(col) is not None:
                try:
                    input_df[f'{col}_encoded'] = label_encoders[col].transform([data[col]])[0]
                except:
                    # Si falla el encoding, usar valor por defecto
                    input_df[f'{col}_encoded'] = 0

        # Asegurar que tengamos todas las columnas del modelo
        for col in columnas_modelo:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reordenar columnas como el modelo espera
        input_df = input_df[columnas_modelo]

        # Escalar variables numéricas
        numeric_cols = ['metros_cuadrados', 'monto_reserva', 'lote_precio_total',
                       'tiempo_reserva_dias', 'SALARIO_APROX', 'n_visitas',
                       'ratio_reserva_precio', 'dias_hasta_limite', 'precio_m2']

        # Filtrar solo las que existen
        numeric_cols = [col for col in numeric_cols if col in input_df.columns]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        return input_df

    except Exception as e:
        st.error(f"Error en preprocesamiento: {e}")
        return None

# Botón de predicción
st.sidebar.markdown("---")
if st.sidebar.button("🎯 Predecir Probabilidad de Compra", type="primary"):

    # Recopilar todos los datos
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
        'SALARIO_APROX': SALARIO_APROX,
        'n_visitas': n_visitas,
        'canal_contacto': canal_contacto,
        'asesor': asesor,
        'promesa_regalo': promesa_regalo,
        'DOCUMENTOS': DOCUMENTOS,
        'CERCA_AVENIDAS': CERCA_AVENIDAS,
        'CERCA_COLEGIOS': CERCA_COLEGIOS,
        'CERCA_PARQUE': CERCA_PARQUE
    }

    # Preprocesar y predecir
    processed_data = preprocess_input(input_data)

    if processed_data is not None:
        try:
            # Hacer predicción
            probabilidad = model.predict_proba(processed_data)[0][1]
            prediccion = model.predict(processed_data)[0]

            # Mostrar resultados
            st.success("✅ Predicción completada!")

            # Mostrar probabilidad con barra de progreso
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric(
                    label="Probabilidad de Compra",
                    value=f"{probabilidad*100:.1f}%"
                )

                if probabilidad > 0.7:
                    st.success("🎉 Alta probabilidad de compra")
                elif probabilidad > 0.4:
                    st.warning("⚠️ Probabilidad media de compra")
                else:
                    st.error("📉 Baja probabilidad de compra")

            with col2:
                st.progress(float(probabilidad))
                st.caption(f"Confianza del modelo: {probabilidad*100:.1f}%")

            # Mostrar detalles de la predicción
            st.subheader("📊 Análisis de la Predicción")

            col3, col4 = st.columns(2)

            with col3:
                st.info("**Factores Positivos:**")
                if monto_reserva == 500:
                    st.write("✅ Monto de reserva alto")
                if n_visitas >= 3:
                    st.write("✅ Múltiples visitas")
                if DOCUMENTOS == 'Completo':
                    st.write("✅ Documentación completa")
                if SALARIO_APROX >= 3000:
                    st.write("✅ Buen nivel de ingresos")
                if CERCA_AVENIDAS == 'Si':
                    st.write("✅ Cerca de avenidas")

            with col4:
                st.warning("**Factores de Riesgo:**")
                if monto_reserva < 300:
                    st.write("❌ Monto de reserva bajo")
                if n_visitas <= 1:
                    st.write("❌ Pocas visitas")
                if DOCUMENTOS == 'Incompleto':
                    st.write("❌ Documentación incompleta")
                if tiempo_reserva_dias > 365:
                    st.write("❌ Tiempo de reserva muy largo")
                if CERCA_COLEGIOS == 'No':
                    st.write("❌ Lejos de colegios")

        except Exception as e:
            st.error(f"Error en la predicción: {e}")

# Información adicional en el main
st.header("📈 Análisis de Clientes")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total de Clientes", "100,000")
    st.metric("Tasa de Conversión", "50%")

with col2:
    st.metric("Mejor Proyecto", "PROYECTO_3")
    st.metric("Asesor Top", "Asesor_15")

with col3:
    st.metric("Canal Más Efectivo", "EVENTO")
    st.metric("Regalo Popular", "Cocina")

st.markdown("---")
st.info("💡 **Recomendaciones:** Para aumentar la probabilidad de compra, considere montos de reserva más altos, documentación completa y seguimiento cercano del asesor.")