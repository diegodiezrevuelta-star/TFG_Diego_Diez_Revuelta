"""
=============================================================================
Predicción de la SED mediante deep learning empleando registros de EEG
Arquitectura Híbrida (CNN-4 + GRU) - Clasificación Binaria 
=============================================================================
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# =============================================================================
#  CONFIGURACIÓN GLOBAL DE ETIQUETA
# =============================================================================
NOMBRE_ETIQUETA = 'epworth'  # Cambiar para las otras etiquetas: mslt, mwt y psqi
ETIQUETA_MIN = 0             # Valor mínimo válido de la etiqueta
ETIQUETA_MAX = 24            # Valor máximo válido de la etiqueta (20 --> mslt y mwt; 21 --> psqi)

# --- Umbral de clasificación binaria ---
# Clase 0 (Sano):    etiqueta <  UMBRAL_SANO
# Clase 1 (No sano): etiqueta >= UMBRAL_SANO
UMBRAL_SANO = 11 # Modificar los umbrales en función de la etiqueta utilizada (mslt-> 8; mwt-> 8; psqi-> 5)
NUM_CLASES  = 2

# =============================================================================
#  CONFIGURACIÓN DE LA ARQUITECTURA POR ÉPOCAS (CNN + GRU)
# =============================================================================
Fs = 100            # Frecuencia de muestreo (Hz)
n_hour = 12         # Horas de registro
nmin = 10           # Duración de cada "época" en minutos
n_sequences = int(n_hour * 60 / nmin)          # Nº de épocas -> 72
insize_per_ep = int(nmin * Fs * 60)            # Puntos por época -> 60000
TARGET_LENGTH = n_sequences * insize_per_ep    # Señal completa -> 4,320,000

print(f"Configuración: n_sequences={n_sequences}, insize_per_ep={insize_per_ep}, TARGET_LENGTH={TARGET_LENGTH}")

# =============================================================================
#  FUNCIONES AUXILIARES
# =============================================================================
#Extrae el valor de la etiqueta configurada para poder estratificar
def obtener_valor_etiqueta(ruta_archivo):
    try:
        meta_data = sio.loadmat(ruta_archivo, variable_names=[NOMBRE_ETIQUETA])
        valor = meta_data[NOMBRE_ETIQUETA]
        while isinstance(valor, (np.ndarray, list)):
            if valor.size == 0: return None
            valor = valor[0]

        if np.isnan(valor) or valor < ETIQUETA_MIN or valor > ETIQUETA_MAX:
            return None
        return valor
    except:
        return None

def valor_a_clase(valor):
    # Convierte la puntuación continua en clase binaria (0=Sano, 1=No sano)
    return 1 if valor >= UMBRAL_SANO else 0 # Modificar en función del tipo de etiqueta

# =============================================================================
#  CONFIGURACIÓN DE RUTAS Y PREPARACIÓN DE DATOS
# =============================================================================
# Carpeta principal donde están guardados los datos
dir_base = os.path.join('.', 'BBDD')
# Lista con todas las bases de datos que queremos procesar
bases_individuales = [r'APPLES\apple_procesado',r'MESA\mesa_procesado', r'MrOS\mros_procesado', r'SHHS\shhs_procesado', r'CFS\cfs_procesado', r'WSC\wsc_procesado']
# Dirección de las bases de datos incluidas en bases_individuales que necesitan ser divididas en subgrupos de train, test y validation.
bases_dividir = [r'APPLES\apple_procesado',r'SHHS\shhs_procesado',r'WSC\wsc_procesado'] 

train_files, test_files, validation_files = [], [], []

# PROCESAMIENTO DE CADA BASE DE DATOS
# Recorremos una por una las carpetas de las bases de datos
for direccion in bases_individuales:
    dir_completa = os.path.join(dir_base, direccion)
    print(f"--- Procesando: {direccion} ---")

    # CASO A:La base de datos necesita ser dividida 
    if direccion in bases_dividir:
        if not os.path.exists(dir_completa): continue

        # Buscamos todos los archivos que terminen en ".mat"
        all_files = [os.path.join(dir_completa, f) for f in os.listdir(dir_completa) if f.endswith('.mat')]
        files_validos, labels = [], []

        # Revisamos archivo por archivo para sacar su "etiqueta"
        for f in all_files:
            val = obtener_valor_etiqueta(f)
            # Si tiene una etiqueta válida, lo guardamos
            if val is not None: 
                files_validos.append(f)
                labels.append(val)

        # Si encontramos archivos válidos empezamos a dividirlos
        if len(files_validos) > 0:
            # Agrupamos las etiquetas en categorías para que la división sea equilibrada
            bins = np.linspace(ETIQUETA_MIN, ETIQUETA_MAX, 6)
            strat_labels = np.digitize(labels, bins)
            
            try:
                # Dividimos las muestras (70% para entrenar, 30% restante)
                tr, temp_files, _, temp_labels_strat = train_test_split(files_validos, strat_labels, test_size=0.30, random_state=42, stratify=strat_labels)
                # Dividimos el 30% de las muestras restantes (15% para test y 15% para validación)
                ts, vl = train_test_split(temp_files, test_size=0.50, random_state=42, stratify=temp_labels_strat)
                # Añadimos los archivos a nuestras listas generales
                train_files.extend(tr)
                test_files.extend(ts)
                validation_files.extend(vl)
                print(f"    Dividido con éxito usando binning: {len(tr)} train, {len(ts)} test, {len(vl)} val")

            except ValueError as e:
                print(f" Error incluso con binning: {e}")
                print("Fallback: Dividiendo sin estratificación para evitar el error.")
                tr, temp = train_test_split(files_validos, test_size=0.30, random_state=42)
                ts, vl = train_test_split(temp, test_size=0.50, random_state=42)
                train_files.extend(tr)
                test_files.extend(ts)
                validation_files.extend(vl)

    # CASO B: La base de datos ya viene dividida
    else:
        # Buscamos directamente en las subcarpetas "train", "test" y "validation"
        for sub, lista in [('train', train_files), ('test', test_files), ('validation', validation_files)]:
            ruta_sub = os.path.join(dir_completa, sub)
            if os.path.exists(ruta_sub):
                # Seleccionamos los archivos ".mat"
                archivos = [os.path.join(ruta_sub, f) for f in os.listdir(ruta_sub) if f.endswith('.mat')]
                # Filtramos para quedarnos solo con los que tienen etiqueta válida
                archivos_validos = [f for f in archivos if obtener_valor_etiqueta(f) is not None]
                # Los añadimos a su lista correspondiente
                lista.extend(archivos_validos)

# --- RESUMEN FINAL ---
print("\n RESULTADO FINAL:")
print(f"Entrenamiento: {len(train_files)} | Test: {len(test_files)} | Val: {len(validation_files)}")

# --- MEDIA Y BALANCE DE CLASES ---
def calcular_media(lista_archivos):
    vals = [obtener_valor_etiqueta(f) for f in lista_archivos]
    return np.mean([v for v in vals if v is not None])

def resumen_clases(lista_archivos, nombre_split):
    # Se extraen los valores numéricos de las etiquetas para cada archivo proporcionado
    vals = [obtener_valor_etiqueta(f) for f in lista_archivos]
    # Se filtran las muestras que carecen de una etiqueta válida
    vals = [v for v in vals if v is not None]
    # Se binarizan las etiquetas continuas
    clases = [valor_a_clase(v) for v in vals]
    # Se contabiliza la frecuencia absoluta de ambas clases
    n_sano = clases.count(0)
    n_no_sano = clases.count(1)
    # Se determina el número total de muestras válidas
    total = len(clases) if len(clases) > 0 else 1
    print(f"   {nombre_split} -> Sano (0): {n_sano} ({100*n_sano/total:.1f}%) | No sano (1): {n_no_sano} ({100*n_no_sano/total:.1f}%)")

print(f"Media {NOMBRE_ETIQUETA} Train: {calcular_media(train_files):.2f}")
print(f"Media {NOMBRE_ETIQUETA} Test:  {calcular_media(test_files):.2f}")
print(f"Media {NOMBRE_ETIQUETA} Val:   {calcular_media(validation_files):.2f}")

print(f"\nBalance de clases (umbral = {UMBRAL_SANO}):")
resumen_clases(train_files, "Train")
resumen_clases(test_files, "Test ")
resumen_clases(validation_files, "Val  ")

# =============================================================================
#  ARQUITECTURA CNN-4
# =============================================================================
def sleepiness_cnn(insize_per_ep):
    
    inputs = layers.Input(shape=(insize_per_ep, 1))

    # ── BLOQUE 1
    x = layers.Conv1D(filters=8, kernel_size=251, strides=10, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(pool_size=8)(x)

    # ── BLOQUE 2
    x = layers.Conv1D(filters=16, kernel_size=71, strides=5, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(0.1)(x)

    # ── BLOQUE 3
    x = layers.Conv1D(filters=32, kernel_size=35, strides=3, padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Dropout(0.1)(x)

    # ── BLOQUE 4
    x = layers.Conv1D(filters=32, kernel_size=17, strides=2, padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(5e-6))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.1)(x)

    # ── Pooling dual 
    gap = layers.GlobalAveragePooling1D()(x)
    gmp = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([gap, gmp])

    # ── Clasificación
    x = layers.Dense(32, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.30)(x)

    return models.Model(inputs=inputs, outputs=x)

def build_complete_sleep_model_binario(n_sequences, insize_per_ep, nunit=64, dropout_rnn=0.3, num_clases=NUM_CLASES):
    
    # Integra la arquitectura CNN con el bloque de redes recurrentes (GRUs) y define la salida para la tarea de regresión.
    inputs = Input(shape=(n_sequences, insize_per_ep, 1))

    # Se instancia el extractor de características base (CNN)
    base_cnn = sleepiness_cnn(insize_per_ep)
    base_cnn.summary()

    # Se envuelve la CNN en una capa TimeDistributed para procesar la secuencia temporal completa
    conv_seq = layers.TimeDistributed(base_cnn)(inputs)
    conv_seq = layers.Dropout(0.3)(conv_seq)

    # Se define el bloque de redes recurrentes (GRUs) para extraer el contexto temporal
    x = layers.GRU(nunit, return_sequences=True, dropout=dropout_rnn, kernel_initializer='he_normal')(conv_seq)
    x = layers.GRU(nunit, return_sequences=True, dropout=dropout_rnn, kernel_initializer='he_normal')(x)
    x = layers.GRU(nunit, return_sequences=False, dropout=dropout_rnn, kernel_initializer='he_normal')(x)

    # Se añade la capa de salida lineal para la predicción del valor  
    out = layers.Dense(num_clases, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=out)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    # Se compila el modelo 
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
    return model

# ── Instanciación del modelo ──────────────────────────────────────────────
model = build_complete_sleep_model_binario(n_sequences, insize_per_ep)
model.summary()

# =============================================================================
#  CALLBACKS
# =============================================================================
early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min', restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
checkpoint = ModelCheckpoint(filepath='modelos_entrenados_2/mejor_modelo_binario.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
# =============================================================================
#  GENERADOR DE DATOS
# =============================================================================
class SignalGeneratorMAT(Sequence):
    
    # Configuración incial               
    def __init__(self, file_paths, batch_size, n_sequences, insize_per_ep, shuffle=True, oversample=False, augment_prob=0.5, noise_std=0.01, max_shift=2000):
        self.original_paths = list(file_paths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.oversample = oversample
        self.n_sequences = n_sequences
        self.insize_per_ep = insize_per_ep
        self.target_length = n_sequences * insize_per_ep
        self.augment_prob = augment_prob
        self.noise_std = noise_std
        self.max_shift = max_shift

        # Se crea un registro  de las clases asociadas a cada archivo.
        self.clase_por_archivo = {}
        for f in self.original_paths:
            val = obtener_valor_etiqueta(f)
            if val is not None:
                self.clase_por_archivo[f] = valor_a_clase(val)

        # Se aplica el balanceo de clases si está habilitado
        if self.oversample:
            self.preparar_oversampling()
        else:
            self.file_paths = np.array(self.original_paths)
            self.is_duplicate = np.zeros(len(self.file_paths), dtype=bool)

        # Se aleatoriza el orden inicial del conjunto de datos
        if self.shuffle:
            self.shuffle_manteniendo_flags()

    def preparar_oversampling(self):
        # Equilibra la distribución de clases replicando aleatoriamente las muestras de la clase minoritaria hasta igualar la cantidad de la clase mayoritaria.
        archivos_por_clase = {0: [], 1: []}
        for f, clase in self.clase_por_archivo.items():
            archivos_por_clase[clase].append(f)

        n_clase0 = len(archivos_por_clase[0])
        n_clase1 = len(archivos_por_clase[1])
        n_mayoria = max(n_clase0, n_clase1)

        balanceado = []
        es_duplicado = []
        # Se iteran las clases para identificar la minoritaria y generar las réplicas necesarias
        for clase, archivos in archivos_por_clase.items():
            if len(archivos) == 0:
                continue
            faltantes = n_mayoria - len(archivos)
            balanceado.extend(archivos)
            es_duplicado.extend([False] * len(archivos))
            # Si existen muestras faltantes para igualar a la mayoría, se realiza un muestreo con reemplazo
            if faltantes > 0:
                extra = list(np.random.choice(archivos, size=faltantes, replace=True))
                balanceado.extend(extra)
                es_duplicado.extend([True] * faltantes)

        self.file_paths = np.array(balanceado)
        self.is_duplicate = np.array(es_duplicado)
        print(f"   [Oversampling] Sano: {n_clase0} -> {n_mayoria} | No sano: {n_clase1} -> {n_mayoria} | Total por época: {len(balanceado)}")

    def shuffle_manteniendo_flags(self):
        # Aleatoriza el conjunto de datos garantizando que cada ruta de archivo conserve su etiqueta indicadora de duplicidad.
        idx = np.random.permutation(len(self.file_paths))
        self.file_paths = self.file_paths[idx]
        self.is_duplicate = self.is_duplicate[idx]

    def __len__(self):
        # Se calcula el número total de batches por época
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def on_epoch_end(self):
        # Al finalizar cada época, se regeneran las réplicas aleatorias y se reordena el conjunto
        if self.oversample:
            self.preparar_oversampling()
        if self.shuffle:
            self.shuffle_manteniendo_flags()

    def augmentar_ligero(self, senal):
        # Aplica transformaciones estocásticas a la señal temporal para mejorar la capacidad de generalización del modelo.
        # Inyección de ruido blanco Gaussiano
        if np.random.rand() < self.augment_prob:
            senal = senal + np.random.normal(0, self.noise_std, size=senal.shape).astype(np.float32)
            # Desplazamiento circular de la señal
        if np.random.rand() < self.augment_prob and self.max_shift > 0:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            senal = np.roll(senal, shift)
        return senal

    def __getitem__(self, idx):
        # Procesa y retorna un batch
        batch_paths = self.file_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_dup = self.is_duplicate[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []

        for path, es_dup in zip(batch_paths, batch_dup):
            try:
                # Se carga la señal
                data = sio.loadmat(path)
                senal = data['senal_final'].flatten().astype(np.float32)

                if len(senal) != self.target_length:
                    senal_procesada = np.zeros(self.target_length, dtype=np.float32)
                    actual_fill = min(len(senal), self.target_length)
                    senal_procesada[:actual_fill] = senal[:actual_fill]
                else:
                    senal_procesada = senal

                # Se aplica el aumento de datos de forma exclusiva a las muestras sintéticas/duplicadas
                if es_dup:
                    senal_procesada = self.augmentar_ligero(senal_procesada)
                    
                    # Se extrae la etiqueta, se clasifica y se codifica en formato One-Hot
                etiqueta_valor = data[NOMBRE_ETIQUETA][0][0]
                clase = valor_a_clase(etiqueta_valor)
                etiqueta_onehot = np.eye(NUM_CLASES, dtype=np.float32)[clase]

                batch_x.append(senal_procesada)
                batch_y.append(etiqueta_onehot)

            except Exception as e:
                print(f"Error procesando {path}: {e}")

        # Se gestiona el error en caso de que el lote resulte vacío
        if len(batch_x) == 0:
            return (np.empty((0, self.n_sequences, self.insize_per_ep, 1), dtype=np.float32), np.empty((0, NUM_CLASES), dtype=np.float32))

        # Se convierten las listas al formato adecuado
        X = np.array(batch_x)
        X = X.reshape((-1, self.n_sequences, self.insize_per_ep, 1))

        return X, np.array(batch_y, dtype=np.float32)

# =============================================================================
#  ENTRENAMIENTO
# =============================================================================
BATCH_SIZE = 8
# Generadores de datos para los conjuntos de entrenamiento y validación
train_gen = SignalGeneratorMAT(train_files, batch_size=BATCH_SIZE, n_sequences=n_sequences, insize_per_ep=insize_per_ep, shuffle=True, oversample=True)
val_gen = SignalGeneratorMAT(validation_files, batch_size=BATCH_SIZE, n_sequences=n_sequences, insize_per_ep=insize_per_ep, shuffle=False, oversample=False)

# Ejecución del bucle de entrenamiento
print("Iniciando entrenamiento...")
history = model.fit(train_gen, validation_data=val_gen if len(validation_files) > 0 else None, epochs=200, verbose=1, callbacks=[early_stop, lr_reducer, checkpoint])

# =============================================================================
#  TEST Y EVALUACIÓN
# =============================================================================
# Generador de datos para el conjunto de prueba 
test_gen = SignalGeneratorMAT(test_files, batch_size=BATCH_SIZE, n_sequences=n_sequences, insize_per_ep=insize_per_ep, shuffle=False)

print("\n--- EVALUACIÓN FORMAL ---")
# Evaluación cuantitativa del modelo utilizando las métricas de pérdida
results = model.evaluate(test_gen, verbose=1)
print(f"Test Loss (Categorical Crossentropy): {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")

print("\nGenerando predicciones detalladas...")
# Obtención de las predicciones del modelo sobre el conjunto de prueba
y_pred_raw = model.predict(test_gen, verbose=1)

# Extracción de las etiquetas reales del generador para la validación estadística
y_true_raw = []
for i in range(len(test_gen)):
    _, labels = test_gen[i]
    y_true_raw.extend(labels)
y_true_raw = np.array(y_true_raw)

# Conversión de one-hot a clase (0 o 1)
y_pred = np.argmax(y_pred_raw, axis=1)
y_true = np.argmax(y_true_raw, axis=1)

# Cálculo de métricas de clasificación
print("\n--- INFORME DE CLASIFICACIÓN ---")
print(classification_report(y_true, y_pred, target_names=['Sano (0)', 'No sano (1)'], zero_division=0))

# Se representa la matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sano (0)', 'No sano (1)'], yticklabels=['Sano (0)', 'No sano (1)'])

plt.title('Matriz de Confusión: Aciertos y Errores')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real')
plt.show()

# Curva ROC y AUC
# Se extraen las probabilidades para la clase 1 ("No sano") adaptándose a la arquitectura
if y_pred_raw.shape[1] == 1:
    y_scores = y_pred_raw.flatten()
else:
    y_scores = y_pred_raw[:, 1]

# Cálculo de la Tasa de Falsos Positivos (FP) y Verdaderos Positivos (TP) a distintos umbrales
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
# Cálculo del Área Bajo la Curva (AUC)
roc_auc = auc(fpr, tpr)
print(f"\n AUC: {roc_auc:.4f}")

# Representación gráfica de la Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC {NOMBRE_ETIQUETA.upper()} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Clasificador Aleatorio')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12)
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12)
plt.title(f'Curva ROC - {NOMBRE_ETIQUETA.upper()}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

# Guardado final del modelo
model.save(f'modelos_entrenados/modelo_cnn4_rnn_{NOMBRE_ETIQUETA}_binario.keras')
