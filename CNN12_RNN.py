"""
=============================================================================
PROYECTO TFG: Predicción de la SED mediante deep learning empleando registros de EEG
Modelo: Arquitectura Híbrida CNN (12 capas) + RNN - Regresión 
=============================================================================
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
#  CONFIGURACIÓN GLOBAL DE ETIQUETA
# =============================================================================
NOMBRE_ETIQUETA = 'epworth'  # Cmbiar para las otras etiquetas: mslt, mwt y psqi
ETIQUETA_MIN = 0             # Valor mínimo válido de la etiqueta
ETIQUETA_MAX = 24            # Valor máximo válido de la etiqueta (20 --> mslt y mwt; 21 --> psqi)

# =============================================================================
#  FUNCIONES AUXILIARES
# =============================================================================
def obtener_valor_etiqueta(ruta_archivo):
    """Extrae el valor de la etiqueta configurada para poder estratificar."""
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

# =============================================================================
#  CONFIGURACIÓN DE RUTAS Y PREPARACIÓN DE DATOS
# =============================================================================
dir_base = os.path.join('.', 'BBDD')
bases_individuales = [r'APPLES\apple_procesado',r'MESA\mesa_procesado', r'MrOS\mros_procesado', r'SHHS\shhs_procesado', r'CFS\cfs_procesado', r'WSC\wsc_procesado'] # Dirección de las bases de datos a utilizar
bases_dividir = [r'APPLES\apple_procesado',r'SHHS\shhs_procesado',r'WSC\wsc_procesado'] # Dirección de las bases de datos incluidas en bases_individuales que necesitan ser divididas en subgrupos de train, test y validation.

train_files, test_files, validation_files = [], [], []

for direccion in bases_individuales:
    dir_completa = os.path.join(dir_base, direccion)
    print(f"--- Procesando: {direccion} ---")

    if direccion in bases_dividir:
        if not os.path.exists(dir_completa): continue

        all_files = [os.path.join(dir_completa, f) for f in os.listdir(dir_completa) if f.endswith('.mat')]
        files_validos, labels = [], []

        for f in all_files:
            val = obtener_valor_etiqueta(f)
            if val is not None:
                files_validos.append(f)
                labels.append(val)

        if len(files_validos) > 0:
            bins = np.linspace(ETIQUETA_MIN, ETIQUETA_MAX, 6)
            strat_labels = np.digitize(labels, bins)
            
            try:
                tr, temp_files, _, temp_labels_strat = train_test_split(files_validos, strat_labels, test_size=0.30, random_state=42, stratify=strat_labels)
                ts, vl = train_test_split(temp_files, test_size=0.50, random_state=42, stratify=temp_labels_strat)
                train_files.extend(tr)
                test_files.extend(ts)
                validation_files.extend(vl)
                print(f" Dividido con éxito usando binning: {len(tr)} train, {len(ts)} test, {len(vl)} val")

            except ValueError as e:
                print(f" Error incluso con binning: {e}")
                print("Fallback: Dividiendo sin estratificación para evitar el error.")
                tr, temp = train_test_split(files_validos, test_size=0.30, random_state=42)
                ts, vl = train_test_split(temp, test_size=0.50, random_state=42)
                train_files.extend(tr)
                test_files.extend(ts)
                validation_files.extend(vl)

    else:
        for sub, lista in [('train', train_files), ('test', test_files), ('validation', validation_files)]:
            ruta_sub = os.path.join(dir_completa, sub)
            if os.path.exists(ruta_sub):
                archivos = [os.path.join(ruta_sub, f) for f in os.listdir(ruta_sub) if f.endswith('.mat')]
                archivos_validos = [f for f in archivos if obtener_valor_etiqueta(f) is not None]
                lista.extend(archivos_validos)

# --- RESUMEN FINAL ---
print("\n RESULTADO FINAL:")
print(f"Entrenamiento: {len(train_files)} | Test: {len(test_files)} | Val: {len(validation_files)}")

def calcular_media(lista_archivos):
    vals = [obtener_valor_etiqueta(f) for f in lista_archivos]
    return np.mean([v for v in vals if v is not None])

print(f"Media {NOMBRE_ETIQUETA} Train: {calcular_media(train_files):.2f}")
print(f"Media {NOMBRE_ETIQUETA} Test:  {calcular_media(test_files):.2f}")
print(f"Media {NOMBRE_ETIQUETA} Val:   {calcular_media(validation_files):.2f}")

# =============================================================================
#  ARQUITECTURA CNN-12
# =============================================================================
def sleepiness_cnn(insize_per_ep):
    model = models.Sequential()

    # Entrada temporal por época
    model.add(layers.Input(shape=(insize_per_ep, 1)))

    # --- BLOQUE 1 ---
    model.add(layers.Conv1D(filters=16, kernel_size=101, strides=5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Conv1D(filters=16, kernel_size=51, strides=4, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Conv1D(filters=16, kernel_size=31, strides=4, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Conv1D(filters=16, kernel_size=31, strides=4, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv1D(filters=16, kernel_size=15, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv1D(filters=16, kernel_size=15, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.1))

    # --- BLOQUE 2 ---
    model.add(layers.Conv1D(filters=32, kernel_size=7, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv1D(filters=32, kernel_size=7, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv1D(filters=32, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv1D(filters=64, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.2))

    model.add(layers.GlobalAveragePooling1D())

    # Capa densa 
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.2))
    
    # Retorna las características (128) para que la GRU haga la regresión.
    return model

# =============================================================================
#  ARQUITECTURA COMPLETA (CNN + GRU)
# =============================================================================
def build_complete_sleep_model(n_sequences, insize_per_ep, nunit=64, dropout_rnn=0.3):
    model = models.Sequential()
    
    # 1. Definir la entrada: (N_Epochs, Puntos_por_Epoch, Canales)
    model.add(layers.Input(shape=(n_sequences, insize_per_ep, 1)))

    # 2. Instanciar el extractor base y envolverlo en TimeDistributed
    base_cnn = sleepiness_cnn(insize_per_ep)
    model.add(layers.TimeDistributed(base_cnn))
    model.add(layers.Dropout(0.3))

    # 3. Stack de GRUs
    model.add(layers.GRU(nunit, return_sequences=True, dropout=dropout_rnn, kernel_initializer='he_normal'))
    model.add(layers.GRU(nunit, return_sequences=True, dropout=dropout_rnn, kernel_initializer='he_normal'))
    model.add(layers.GRU(nunit, return_sequences=False, dropout=dropout_rnn, kernel_initializer='he_normal'))

    # 4. Capa de salida: Regresión
    model.add(layers.Dense(1, activation="linear", kernel_initializer='he_normal'))

    # 5. Compilación del modelo
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='huber', optimizer=opt, metrics=['mae'])
    
    return model

# --- Configuración de parámetros de la señal ---
Fs = 100            # Frecuencia de muestreo (Hz)
n_hour = 12         # Horas de registro
nmin = 10           # Duración de cada "época" en minutos
n_sequences = int(n_hour * 60 / nmin)
insize_per_ep = int(nmin * Fs * 60)

model = build_complete_sleep_model(n_sequences, insize_per_ep)
model.summary()

# =============================================================================
#  GENERADOR DE DATOS
# =============================================================================
class SignalGeneratorMAT(Sequence):
    def __init__(self, file_paths, batch_size, shuffle=True):
        self.file_paths = np.array(file_paths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_paths)

    def __getitem__(self, idx):
        batch_paths = self.file_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x, batch_y = [], []

        for path in batch_paths:
            try:
                data = sio.loadmat(path)
                senal = data['senal_final'].flatten()

                etiqueta = data[NOMBRE_ETIQUETA].flatten()[0]
                senal_2 = senal.reshape(n_sequences, insize_per_ep, 1)

                batch_x.append(senal_2)
                batch_y.append(etiqueta)
            except Exception as e:
                print(f"Error cargando {path}: {e}")

        X = np.array(batch_x)
        return X, np.array(batch_y)


# =============================================================================
#  ENTRENAMIENTO
# =============================================================================
BATCH_SIZE = 8

train_gen = SignalGeneratorMAT(train_files, batch_size=BATCH_SIZE, shuffle=True)
val_gen = SignalGeneratorMAT(validation_files, batch_size=BATCH_SIZE, shuffle=False)

early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min', restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)

os.makedirs('modelos_entrenados', exist_ok=True)
ruta_checkpoint = os.path.join('modelos_entrenados', f'mejor_modelo_{NOMBRE_ETIQUETA}.keras')

checkpoint = ModelCheckpoint(filepath=ruta_checkpoint, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

print("Iniciando entrenamiento...")
history = model.fit(train_gen,validation_data=val_gen if len(validation_files) > 0 else None,epochs=200,verbose=1,callbacks=[early_stop, lr_reducer, checkpoint])

# =============================================================================
#  EVALUACIÓN Y GRÁFICAS
# =============================================================================
test_gen = SignalGeneratorMAT(test_files, batch_size=BATCH_SIZE, shuffle=False)

print("\n--- EVALUACIÓN FORMAL (Huber/MAE) ---")
results = model.evaluate(test_gen, verbose=1)
print(f"Test Loss (Huber): {results[0]:.4f}")
print(f"Test MAE: {results[1]:.4f}")

print("\nGenerando predicciones detalladas...")
y_pred = model.predict(test_gen, verbose=1).flatten()

y_true = []
for i in range(len(test_gen)):
    _, labels = test_gen[i]
    y_true.extend(labels)
y_true = np.array(y_true)

r_pearson, p_value = pearsonr(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n--- MÉTRICAS DE CORRELACIÓN Y PRECISIÓN ---")
print(f"Correlación de Pearson: {r_pearson:.4f} (p-value: {p_value:.4e})")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# Dibujar ajuste
m, b = np.polyfit(y_true, y_pred, 1)

plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.4, color='blue', label='Pacientes')

x_range = np.array([ETIQUETA_MIN, ETIQUETA_MAX + 4])
plt.plot(x_range, m * x_range + b, color='green', linestyle='-', linewidth=2, label=f'Ajuste Real ($y = {m:.2f}x + {b:.2f}$)')

plt.title(f'Análisis de Sesgo: Predicción vs Realidad ({NOMBRE_ETIQUETA.upper()})')
plt.xlabel(f'Valores Reales ({NOMBRE_ETIQUETA})')
plt.ylabel('Predicciones del Modelo')
plt.xlim(ETIQUETA_MIN, ETIQUETA_MAX + 4)
plt.ylim(ETIQUETA_MIN, ETIQUETA_MAX + 4)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print(f"Ecuación de la recta de ajuste: y = {m:.3f}x + {b:.3f}")

# Guardado final del modelo
model.save(f'modelos_entrenados/modelo_cnn12rnn_{NOMBRE_ETIQUETA}.keras')
