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

# =============================
# CARGA DEL MODELO
# =============================
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
        st.error(f"Error al cargar el modelo: {e}")
        return None, None, None, None


model, scaler, columnas_modelo, label_encoders = load_model()

if model is None:
    st.stop()

# =============================
# INTERFAZ PRINCIPAL CON PESTAÑAS
# =============================
tab_prediccion, tab_historial = st.tabs(["🧠 Predicción", "📜 Historial de Evaluaciones"])

# ============================================================
# 🧠 SECCIÓN 1: PESTAÑA DE PREDICCIÓN
# ============================================================
with tab_prediccion:

    st.title("🏠 Predictor de Probabilidad de Compra - Inmobiliaria")
    st.markdown("---")

    # ======== ENTRADA DE DATOS ========
    st.sidebar.header("📋 Datos del Cliente y Propiedad")

    # DNI
    dni_cliente = st.sidebar.text_input("DNI del Cliente", max_chars=8)

    st.sidebar.subheader("1. Información del Proyecto")
    proyecto = st.sidebar.selectbox("Proyecto",
        ['PROYECTO_1','PROYECTO_2','PROYECTO_3','PROYECTO_4','PROYECTO_5',
         'PROYECTO_6','PROYECTO_7','PROYECTO_8','PROYECTO_9','PROYECTO_10'])

    manzana = st.sidebar.selectbox("Manzana", ['Mz-A','Mz-B','Mz-C','Mz-D','Mz-E'])

    lote_ubicacion = st.sidebar.selectbox("Ubicación del Lote",
        ['UBICACION_1','UBICACION_2','UBICACION_3','UBICACION_4','UBICACION_5',
         'UBICACION_6','UBICACION_7','UBICACION_8','UBICACION_9','UBICACION_10'])

    st.sidebar.subheader("2. Características del Lote")
    metros_cuadrados = st.sidebar.slider("Metros Cuadrados",80,200,140,5)
    lote_precio_total = st.sidebar.selectbox("Precio Total del Lote ($)",
        list(range(15000,41000,1000)))

    st.sidebar.subheader("3. Información de Reserva")
    monto_reserva = st.sidebar.selectbox("Monto de Reserva ($)", [500,600,800,900,1000,2000,5000,10000])
    tiempo_reserva_dias = st.sidebar.slider("Tiempo de Reserva (días)",1,90,7,1)
    dias_hasta_limite = st.sidebar.slider("Días hasta Fecha Límite",1,30,30,1)
    metodo_pago = st.sidebar.selectbox("Método de Pago",['EFECTIVO','TARJETA','YAPE'])

    st.sidebar.subheader("4. Información del Cliente")
    cliente_edad = st.sidebar.slider("Edad del Cliente",25,70,45,1)
    cliente_genero = st.sidebar.selectbox("Género del Cliente",['M','F'])
    cliente_profesion = st.sidebar.selectbox("Profesión del Cliente",
        ['Ingeniero','Doctor','Abogado','Docente','Comerciante','Empresario','Otro'])
    cliente_distrito = st.sidebar.selectbox("Distrito del Cliente",
        ['Distrito_A','Distrito_B','Distrito_C','Distrito_D','Distrito_E'])
    SALARIO_DECLARADO = st.sidebar.selectbox("Salario Aproximado ($)",
        [1000,1500,2000,2500,3000,3500,4000,4500,5000])

    st.sidebar.subheader("5. Comportamiento y Características")
    n_visitas = st.sidebar.slider("Número de Visitas",0,5,2,1)
    canal_contacto = st.sidebar.selectbox("Canal de Contacto",
        ['EVENTO','FACEBOOK','PAGINA WEB','WHATSAPP','INSTAGRAM','VOLANTES'])
    asesor = st.sidebar.selectbox("Asesor",[f"Asesor_{i}" for i in range(1,101)])
    promesa_regalo = st.sidebar.selectbox("Promesa de Regalo",
        ['Ninguno','Cocina','Refrigeradora','TV','Lavadora'])
    DOCUMENTOS = st.sidebar.selectbox("Estado de Documentos",
        ['Completo','Incompleto','Pendiente'])

    st.sidebar.subheader("6. Ubicación y Amenities")
    CERCA_AVENIDAS = st.sidebar.selectbox("Cerca de Avenidas",['Si','No'])
    CERCA_COLEGIOS = st.sidebar.selectbox("Cerca de Colegios",['Si','No'])
    CERCA_PARQUE = st.sidebar.selectbox("Cerca de Parques",['Si','No'])

    # =============================
    # FUNCIÓN DE PREPROCESAMIENTO
    # =============================
    def preprocess_input(data):
        try:
            input_df = pd.DataFrame([data])

            # Nuevas variables derivadas
            input_df['ratio_reserva_precio'] = input_df['monto_reserva'] / input_df['lote_precio_total']
            input_df['precio_m2'] = input_df['lote_precio_total'] / input_df['metros_cuadrados']

            # Codificación edad
            input_df['cliente_edad_cat_36-45'] = int(36 <= data['cliente_edad'] <= 45)
            input_df['cliente_edad_cat_46-55'] = int(46 <= data['cliente_edad'] <= 55)
            input_df['cliente_edad_cat_56-70'] = int(data['cliente_edad'] >= 56)

            # One-hot manual
            categorical_mappings = {
                'metodo_pago': ['EFECTIVO','TARJETA','YAPE'],
                'cliente_genero': ['M','F'],
                'cliente_profesion': ['Ingeniero','Doctor','Abogado','Docente','Comerciante','Empresario','Otro'],
                'cliente_distrito': ['Distrito_A','Distrito_B','Distrito_C','Distrito_D','Distrito_E'],
                'canal_contacto': ['EVENTO','FACEBOOK','PAGINA WEB','WHATSAPP','INSTAGRAM','VOLANTES'],
                'promesa_regalo': ['Ninguno','Cocina','Refrigeradora','TV','Lavadora'],
                'DOCUMENTOS': ['Completo','Incompleto','Pendiente'],
                'CERCA_AVENIDAS': ['Si','No'],
                'CERCA_COLEGIOS': ['Si','No'],
                'CERCA_PARQUE': ['Si','No']
            }

            for col, values in categorical_mappings.items():
                for value in values[1:]:
                    col_name = f"{col}_{value}"
                    input_df[col_name] = 1 if data[col] == value else 0

            # Label Encoding para columnas con muchos valores
            for col in ['proyecto','manzana','asesor','lote_ubicacion']:
                if label_encoders.get(col):
                    try:
                        input_df[f"{col}_encoded"] = label_encoders[col].transform([data[col]])[0]
                    except:
                        input_df[f"{col}_encoded"] = 0

            # Alinear columnas
            for col in columnas_modelo:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[columnas_modelo]

            # Escalado
            numeric_cols = ['metros_cuadrados','monto_reserva','lote_precio_total',
                            'tiempo_reserva_dias','SALARIO_DECLARADO','n_visitas',
                            'ratio_reserva_precio','dias_hasta_limite','precio_m2']
            numeric_cols = [c for c in numeric_cols if c in input_df.columns]
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

            return input_df
        except Exception as e:
            st.error(f"Error en preprocesamiento: {e}")
            return None

    # =============================
    # BOTÓN DE PREDICCIÓN
    # =============================
    if st.sidebar.button("🎯 Predecir Probabilidad de Compra", type="primary"):
        input_data = {
            'dni_cliente': dni_cliente,
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
                prob = model.predict_proba(processed_data)[0][1]
                pred = model.predict(processed_data)[0]

                st.success("✅ Predicción completada")
                st.metric("Probabilidad de Compra", f"{prob*100:.2f}%")
                st.progress(float(prob))

                registro = input_data.copy()
                registro['probabilidad'] = round(prob*100, 2)
                registro['prediccion'] = int(pred)

                archivo_csv = "evaluaciones_clientes.csv"

                # ====== Botón Guardar ======
                if st.button("💾 Guardar Predicción"):
                    df_registro = pd.DataFrame([registro])
                    if os.path.exists(archivo_csv):
                        df_registro.to_csv(archivo_csv, mode='a', header=False, index=False)
                    else:
                        df_registro.to_csv(archivo_csv, index=False)
                    st.success("📁 Predicción guardada correctamente")

            except Exception as e:
                st.error(f"Error en la predicción: {e}")

# ============================================================
# 📜 SECCIÓN 2: PESTAÑA DE HISTORIAL
# ============================================================
with tab_historial:
    st.title("📜 Historial de Evaluaciones de Clientes")
    archivo_csv = "evaluaciones_clientes.csv"

    if os.path.exists(archivo_csv):
        df = pd.read_csv(archivo_csv)
        st.dataframe(df.tail(20))

        st.download_button(
            label="⬇️ Descargar Historial Completo",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="evaluaciones_clientes.csv",
            mime="text/csv"
        )
    else:
        st.info("Aún no hay predicciones guardadas.")
