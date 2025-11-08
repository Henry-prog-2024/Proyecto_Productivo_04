import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# -------------------------
# Configuración de la página
# -------------------------
st.set_page_config(
    page_title="Predictor de Compras - Inmobiliaria",
    page_icon="🏠",
    layout="wide"
)

# Título principal
st.title("🏠 Predictor de Probabilidad de Compra - Inmobiliaria")
st.markdown("---")

# Inicializar historial en sesión
if "historial" not in st.session_state:
    st.session_state["historial"] = None  # lo almacenamos también en CSV, session mantiene referencia reciente
if "ultima_pred" not in st.session_state:
    st.session_state["ultima_pred"] = None

# -------------------------
# Cargar modelo y recursos
# -------------------------
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

model, scaler, columnas_modelo, label_encoders = load_model()
if model is None:
    st.stop()

# -------------------------
# Interfaz: pestañas Predicción / Historial
# -------------------------
tab_pred, tab_hist = st.tabs(["🔮 Predicción", "📜 Historial"])

# ---------- SIDEBAR (mismo formulario base) ----------
with tab_pred:
    st.sidebar.header("📋 Datos del Cliente y Propiedad (Formulario base)")

    # DNI (nuevo campo)
    dni_cliente = st.sidebar.text_input("🪪 DNI del Cliente", max_chars=12, key="dni_input")

    st.sidebar.subheader("1. Información del Proyecto")
    proyecto = st.sidebar.selectbox(
        "Proyecto",
        ['PROYECTO_1', 'PROYECTO_2', 'PROYECTO_3', 'PROYECTO_4', 'PROYECTO_5',
         'PROYECTO_6', 'PROYECTO_7', 'PROYECTO_8', 'PROYECTO_9', 'PROYECTO_10'],
        key="proyecto"
    )

    manzana = st.sidebar.selectbox(
        "Manzana",
        ['Mz-A', 'Mz-B', 'Mz-C', 'Mz-D', 'Mz-E'],
        key="manzana"
    )

    lote_ubicacion = st.sidebar.selectbox(
        "Ubicación del Lote",
        ['UBICACION_1', 'UBICACION_2', 'UBICACION_3', 'UBICACION_4', 'UBICACION_5',
         'UBICACION_6', 'UBICACION_7', 'UBICACION_8', 'UBICACION_9', 'UBICACION_10'],
        key="lote_ubicacion"
    )

    st.sidebar.subheader("2. Características del Lote")
    metros_cuadrados = st.sidebar.slider("Metros Cuadrados", 80, 200, 140, 5, key="metros_cuadrados")
    lote_precio_total = st.sidebar.selectbox("Precio Total del Lote ($)", [15000, 16000, 17000, 18000, 19000, 20000,
                                                                           21000, 22000, 23000, 24000, 25000, 26000,
                                                                           27000, 28000, 29000, 30000, 31000, 32000,
                                                                           33000, 34000, 35000, 36000, 37000, 38000,
                                                                           39000, 40000], key="lote_precio_total")

    st.sidebar.subheader("3. Información de Reserva")
    monto_reserva = st.sidebar.selectbox("Monto de Reserva ($)", [500, 600, 800, 900, 1000, 2000, 5000, 10000], key="monto_reserva")
    tiempo_reserva_dias = st.sidebar.slider("Tiempo de Reserva (días)", 1, 90, 7, 1, key="tiempo_reserva_dias")
    dias_hasta_limite = st.sidebar.slider("Días hasta Fecha Límite", 1, 90, 30, 1, key="dias_hasta_limite")
    metodo_pago = st.sidebar.selectbox("Método de Pago", ['EFECTIVO', 'TARJETA', 'YAPE'], key="metodo_pago")

    st.sidebar.subheader("4. Información del Cliente")
    cliente_edad = st.sidebar.slider("Edad del Cliente", 25, 70, 45, 1, key="cliente_edad")
    cliente_genero = st.sidebar.selectbox("Género del Cliente", ['M', 'F'], key="cliente_genero")
    cliente_profesion = st.sidebar.selectbox("Profesión del Cliente",
                                             ['Ingeniero', 'Doctor', 'Abogado', 'Docente', 'Comerciante', 'Empresario', 'Otro'],
                                             key="cliente_profesion")
    cliente_distrito = st.sidebar.selectbox("Distrito del Cliente",
                                            ['Distrito_A', 'Distrito_B', 'Distrito_C', 'Distrito_D', 'Distrito_E'],
                                            key="cliente_distrito")
    SALARIO_DECLARADO = st.sidebar.selectbox("Salario Aproximado ($)",
                                             [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
                                             key="SALARIO_DECLARADO")

    st.sidebar.subheader("5. Comportamiento y Características")
    n_visitas = st.sidebar.slider("Número de Visitas", 0, 5, 2, 1, key="n_visitas")
    canal_contacto = st.sidebar.selectbox("Canal de Contacto",
                                          ['EVENTO', 'FACEBOOK', 'PAGINA WEB', 'WHATSAPP', 'INSTAGRAM', 'VOLANTES'],
                                          key="canal_contacto")
    asesor = st.sidebar.selectbox("Asesor", [f"Asesor_{i}" for i in range(1, 101)], key="asesor")
    promesa_regalo = st.sidebar.selectbox("Promesa de Regalo",
                                          ['Ninguno', 'Cocina', 'Refrigeradora', 'TV', 'Lavadora'],
                                          key="promesa_regalo")
    DOCUMENTOS = st.sidebar.selectbox("Estado de Documentos", ['Completo', 'Incompleto', 'Pendiente'], key="DOCUMENTOS")

    st.sidebar.subheader("6. Ubicación y Amenities")
    CERCA_AVENIDAS = st.sidebar.selectbox("Cerca de Avenidas", ['Si', 'No'], key="CERCA_AVENIDAS")
    CERCA_COLEGIOS = st.sidebar.selectbox("Cerca de Colegios", ['Si', 'No'], key="CERCA_COLEGIOS")
    CERCA_PARQUE = st.sidebar.selectbox("Cerca de Parques", ['Si', 'No'], key="CERCA_PARQUE")

    # ---------- Preprocessing function (same logic as base) ----------
    def preprocess_input(data: dict):
        try:
            input_df = pd.DataFrame([data])

            # feature engineering
            input_df['ratio_reserva_precio'] = input_df['monto_reserva'] / input_df['lote_precio_total']
            input_df['precio_m2'] = input_df['lote_precio_total'] / input_df['metros_cuadrados']

            # edad categorizada (mantengo tu lógica)
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

            # one-hot manual (mantengo tu mapeo)
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

            # label encoding
            for col in ['proyecto', 'manzana', 'asesor', 'lote_ubicacion']:
                enc = label_encoders.get(col)
                if enc is not None:
                    try:
                        input_df[f'{col}_encoded'] = enc.transform([data[col]])[0]
                    except:
                        input_df[f'{col}_encoded'] = 0
                else:
                    input_df[f'{col}_encoded'] = 0

            # Ensure all model columns are present
            for col in columnas_modelo:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[columnas_modelo]

            # scale numeric columns that exist
            numeric_cols = ['metros_cuadrados', 'monto_reserva', 'lote_precio_total',
                            'tiempo_reserva_dias', 'SALARIO_DECLARADO', 'n_visitas',
                            'ratio_reserva_precio', 'dias_hasta_limite', 'precio_m2']
            numeric_cols = [c for c in numeric_cols if c in input_df.columns]

            # safe transform: check shapes
            if len(numeric_cols) > 0:
                input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

            return input_df

        except Exception as e:
            st.error(f"Error en preprocesamiento: {e}")
            return None

    # ---------- Botón de predicción (único flujo) ----------
    st.sidebar.markdown("---")
    if st.sidebar.button("🎯 Predecir Probabilidad de Compra", key="run_predict"):

        # Construir el diccionario de entrada (incluye DNI)
        input_data = {
            'dni_cliente': dni_cliente,
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

        processed = preprocess_input(input_data)
        if processed is None:
            st.error("Preprocesamiento falló — revisa las columnas y el scaler.")
        else:
            try:
                probabilidad = model.predict_proba(processed)[0][1]
                prediccion = int(model.predict(processed)[0])

                # Mostrar resultado (igual que en tu bloque original)
                st.success("✅ Predicción completada!")
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

                st.subheader("📊 Análisis de la Predicción")
                col3, col4 = st.columns(2)
                with col3:
                    st.info("**Factores Positivos:**")
                    if monto_reserva >= 2000: st.write("✅ Monto de reserva alto")
                    if n_visitas >= 3: st.write("✅ Múltiples visitas")
                    if DOCUMENTOS == 'Completo': st.write("✅ Documentación completa")
                    if SALARIO_DECLARADO >= 3000: st.write("✅ Buen nivel de ingresos")
                    if tiempo_reserva_dias <= 30: st.write("✅ Tiempo de reserva promedio")
                    if CERCA_AVENIDAS == 'Si': st.write("✅ Cerca de avenidas")
                with col4:
                    st.warning("**Factores de Riesgo:**")
                    if monto_reserva < 1000: st.write("❌ Monto de reserva bajo")
                    if monto_reserva >= 1000 and monto_reserva < 2000: st.write("❌ Monto de reserva medio")
                    if n_visitas <= 2: st.write("❌ Pocas visitas")
                    if DOCUMENTOS == 'Incompleto': st.write("❌ Documentación incompleta")
                    if tiempo_reserva_dias >= 31: st.write("❌ Tiempo de reserva muy largo")
                    if SALARIO_DECLARADO < 3000: st.write("❌ Bajo nivel de ingresos")

                # Guardar en session_state la última predicción para el botón de guardar
                registro = input_data.copy()
                registro['probabilidad'] = round(probabilidad*100, 2)
                registro['prediccion'] = prediccion
                registro['fecha'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["ultima_pred"] = registro

                # Botón guardar (con key único)
                st.markdown("---")
                st.subheader("💾 Guardar Resultados de Evaluación")

                if st.button("💾 Guardar Evaluación", key="guardar_evaluacion"):
                    archivo_csv = "evaluaciones_clientes.csv"
                    df_registro = pd.DataFrame([st.session_state["ultima_pred"]])

                    # Si no existe archivo, guardar con header; si existe, append sin header
                    if os.path.exists(archivo_csv):
                        df_registro.to_csv(archivo_csv, mode='a', header=False, index=False)
                    else:
                        df_registro.to_csv(archivo_csv, index=False)

                    st.success("✅ Evaluación guardada correctamente en 'evaluaciones_clientes.csv'")

                    # actualizar session historial para visualización rápida
                    try:
                        df_hist = pd.read_csv(archivo_csv)
                        st.session_state["historial"] = df_hist
                    except:
                        st.session_state["historial"] = df_registro

                # Mostrar historial reciente (si existe)
                archivo_csv = "evaluaciones_clientes.csv"
                if os.path.exists(archivo_csv):
                    df_historial = pd.read_csv(archivo_csv)
                    st.subheader("📂 Historial de Evaluaciones Recientes")
                    st.dataframe(df_historial.tail(10))
                    csv_bytes = df_historial.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Descargar Historial Completo",
                        data=csv_bytes,
                        file_name="evaluaciones_clientes.csv",
                        mime="text/csv",
                        key="download_hist"
                    )

            except Exception as e:
                st.error(f"Error en la predicción: {e}")

# ---------- Pestaña Historial explícita ----------
with tab_hist:
    st.header("📜 Historial de Evaluaciones")
    archivo_csv = "evaluaciones_clientes.csv"
    if os.path.exists(archivo_csv):
        df_hist = pd.read_csv(archivo_csv)
        st.dataframe(df_hist.sort_values(by="fecha", ascending=False) if "fecha" in df_hist.columns else df_hist.tail(100))
        csv_bytes = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Descargar Historial Completo", data=csv_bytes, file_name="evaluaciones_clientes.csv", mime="text/csv", key="download_hist_tab")
    else:
        st.info("Aún no hay predicciones guardadas.")

# ---------------------------
# Sección footer (información original)
# ---------------------------
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
st.info("💡 **Recomendación:** Prioriza montos de reserva altos, documentación completa y seguimiento personalizado para mejorar la probabilidad de compra.")
