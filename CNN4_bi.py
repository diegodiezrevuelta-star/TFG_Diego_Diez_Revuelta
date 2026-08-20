"""
=============================================================================
PROYECTO TFG: Predicción de la SED mediante deep learning empleando registros de EEG
Modelo: Arquitectura CNN (4 capas) - Clasificación Binaria PSQI
=============================================================================
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

 =============================================================================
#  CONFIGURACIÓN GLOBAL DE ETIQUETA
# =============================================================================
NOMBRE_ETIQUETA = 'epworth'  # Cmbiar para las otras etiquetas: mslt, mwt y psqi
ETIQUETA_MIN = 0             # Valor mínimo válido de la etiqueta
ETIQUETA_MAX = 24            # Valor máximo válido de la etiqueta (20 --> mslt y mwt; 21 --> psqi)

# --- Umbral de clasificación binaria ---
# Clase 0 (Sano)
# Clase 1 (No sano)
UMBRAL_SANO = 11 # Modificar los umbrales en función de la etiqueta utilizada (mslt-> 8; mwt-> 8; psqi-> 5)
NUM_CLASES  = 2

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

def valor_a_clase(valor):
    """Convierte la puntuación continua en clase binaria (0=Sano, 1=No sano)."""
    return 1 if valor >= UMBRAL_SANO else 0 # Modificar en función del tipo de etiqueta

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

# --- MEDIA Y BALANCE DE CLASES ---
def calcular_media(lista_archivos):
    vals = [obtener_valor_etiqueta(f) for f in lista_archivos]
    return np.mean([v for v in vals if v is not None])

def resumen_clases(lista_archivos, nombre_split):
    vals = [obtener_valor_etiqueta(f) for f in lista_archivos]
    vals = [v for v in vals if v is not None]
    clases = [valor_a_clase(v) for v in vals]
    n_sano = clases.count(0)
    n_no_sano = clases.count(1)
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
#  ARQUITECTURA DEL MODELO (CNN-4)
# =============================================================================
def sleepiness_cnn(input_shape=(4320000, 1), num_classes=NUM_CLASES):
    # Declaramos el tensor de entrada
    inputs = layers.Input(shape=input_shape)

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
    x = layers.Conv1D(filters=32, kernel_size=35, strides=3, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Dropout(0.1)(x)

    # ── BLOQUE 4
    x = layers.Conv1D(filters=32, kernel_size=17, strides=2, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-6))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.1)(x)

    # ── Pooling dual ──
    gap = layers.GlobalAveragePooling1D()(x)
    gmp = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([gap, gmp])

    # ── Clasificación
    x = layers.Dense(32, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.25)(x)

    # Salida 
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

# ── Compilación ───────────────────────────────────────────────────────────────
model = sleepiness_cnn()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

# =============================================================================
#  CALLBACKS
# =============================================================================
early_stop = EarlyStopping(monitor='val_loss',patience=8,verbose=1,mode='min',restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5,min_lr=1e-7,verbose=1)
checkpoint = ModelCheckpoint(filepath='mejor_modelo_binario.keras',monitor='val_loss',save_best_only=True,mode='min',verbose=1)


# =============================================================================
#  GENERADOR DE DATOS (Con Oversampling)
# =============================================================================
class SignalGeneratorMAT(Sequence):
    def __init__(self, file_paths, batch_size, shuffle=True, oversample=False, augment_prob=0.5, noise_std=0.01, max_shift=2000):
        self.original_paths = list(file_paths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.oversample = oversample
        self.target_length = 4320000
        self.augment_prob = augment_prob
        self.noise_std = noise_std
        self.max_shift = max_shift

        # Cache de etiquetas
        self.clase_por_archivo = {}
        for f in self.original_paths:
            val = obtener_valor_etiqueta(f)
            if val is not None:
                self.clase_por_archivo[f] = valor_a_clase(val)

        if self.oversample:
            self.preparar_oversampling()
        else:
            self.file_paths = np.array(self.original_paths)
            self.is_duplicate = np.zeros(len(self.file_paths), dtype=bool)

        if self.shuffle:
            self.shuffle_manteniendo_flags()

    def preparar_oversampling(self):
        archivos_por_clase = {0: [], 1: []}
        for f, clase in self.clase_por_archivo.items():
            archivos_por_clase[clase].append(f)

        n_clase0 = len(archivos_por_clase[0])
        n_clase1 = len(archivos_por_clase[1])
        n_mayoria = max(n_clase0, n_clase1)

        balanceado = []
        es_duplicado = []
        for clase, archivos in archivos_por_clase.items():
            if len(archivos) == 0:
                continue
            faltantes = n_mayoria - len(archivos)
            balanceado.extend(archivos)
            es_duplicado.extend([False] * len(archivos))
            if faltantes > 0:
                extra = list(np.random.choice(archivos, size=faltantes, replace=True))
                balanceado.extend(extra)
                es_duplicado.extend([True] * faltantes)

        self.file_paths = np.array(balanceado)
        self.is_duplicate = np.array(es_duplicado)
        print(f" [Oversampling] Sano: {n_clase0} -> {n_mayoria} | No sano: {n_clase1} -> {n_mayoria} | Total por época: {len(balanceado)}")

    def shuffle_manteniendo_flags(self):
        idx = np.random.permutation(len(self.file_paths))
        self.file_paths = self.file_paths[idx]
        self.is_duplicate = self.is_duplicate[idx]

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.oversample:
            self.preparar_oversampling()
        if self.shuffle:
            self.shuffle_manteniendo_flags()

    def augmentar_ligero(self, senal):
        if np.random.rand() < self.augment_prob:
            senal = senal + np.random.normal(0, self.noise_std, size=senal.shape).astype(np.float32)
        if np.random.rand() < self.augment_prob and self.max_shift > 0:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            senal = np.roll(senal, shift)
        return senal

    def __getitem__(self, idx):
        batch_paths = self.file_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_dup = self.is_duplicate[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []

        for path, es_dup in zip(batch_paths, batch_dup):
            try:
                data = sio.loadmat(path)
                senal = data['senal_final'].flatten().astype(np.float32)

                if len(senal) != self.target_length:
                    senal_procesada = np.zeros(self.target_length, dtype=np.float32)
                    actual_fill = min(len(senal), self.target_length)
                    senal_procesada[:actual_fill] = senal[:actual_fill]
                else:
                    senal_procesada = senal

                if es_dup:
                    senal_procesada = self.augmentar_ligero(senal_procesada)

                etiqueta_valor = data[NOMBRE_ETIQUETA][0][0]
                clase = valor_a_clase(etiqueta_valor)
                etiqueta_onehot = np.eye(NUM_CLASES, dtype=np.float32)[clase]

                batch_x.append(senal_procesada)
                batch_y.append(etiqueta_onehot)

            except Exception as e:
                print(f"Error procesando {path}: {e}")

        if len(batch_x) == 0:
            return (np.empty((0, self.target_length, 1), dtype=np.float32), np.empty((0, NUM_CLASES), dtype=np.float32))

        X = np.array(batch_x)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=-1)

        return X, np.array(batch_y, dtype=np.float32)
      

# =============================================================================
#  ENTRENAMIENTO
# =============================================================================
BATCH_SIZE = 8

train_gen = SignalGeneratorMAT(train_files, batch_size=BATCH_SIZE, shuffle=True, oversample=True)
val_gen = SignalGeneratorMAT(validation_files, batch_size=BATCH_SIZE, shuffle=False, oversample=False)

print("Iniciando entrenamiento...")
history = model.fit(train_gen,validation_data=val_gen if len(validation_files) > 0 else None,epochs=200,verbose=1,callbacks=[early_stop, lr_reducer, checkpoint])

# =============================================================================
#  TEST Y EVALUACIÓN
# =============================================================================
test_gen = SignalGeneratorMAT(test_files, batch_size=BATCH_SIZE, shuffle=False)

print("\n--- EVALUACIÓN FORMAL ---")
results = model.evaluate(test_gen, verbose=1)
print(f"Test Loss (Categorical Crossentropy): {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")

print("\nGenerando predicciones detalladas...")
y_pred_raw = model.predict(test_gen, verbose=1)

y_true_raw = []
for i in range(len(test_gen)):
    _, labels = test_gen[i]
    y_true_raw.extend(labels)
y_true_raw = np.array(y_true_raw)

# --- CONVERSIÓN DE ONE-HOT A CLASE (0 o 1) ---
y_pred = np.argmax(y_pred_raw, axis=1)
y_true = np.argmax(y_true_raw, axis=1)

# Cálculo de métricas de clasificación
print("\n--- INFORME DE CLASIFICACIÓN ---")
print(classification_report(y_true, y_pred, target_names=['Sano (0)', 'No sano (1)'], zero_division=0))

# Dibujar la Matriz de Confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sano (0)', 'No sano (1)'],
            yticklabels=['Sano (0)', 'No sano (1)'])

plt.title('Matriz de Confusión: Aciertos y Errores')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real')
plt.show()

# Cálculo opcional de Accuracy manual para verificar
acc_manual = accuracy_score(y_true, y_pred)
print(f"\nAccuracy final verificado: {acc_manual:.4f}")

# Guardado final del modelo
model.save(f'modelos_entrenados/modelo_cnn4_{NOMBRE_ETIQUETA}_binario.keras')
