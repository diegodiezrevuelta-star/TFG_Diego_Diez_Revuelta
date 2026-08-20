%% =========================================================================
%  PROYECTO TFG: Procesamiento y Detección de Fin de Registro en Señales EEG
%  Descripción:
%     Este script analiza registros de EEG en formato .mat para detectar
%     automáticamente la desconexión de electrodos o fin de prueba en señales 
%     largas (>12 h) mediante el cálculo de la potencia RMS por bloques y el
%     rango intercuartílico (IQR). Finalmente, genera las figuras de corte.
% =========================================================================

clear; clc; close all;

%% 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
% Define el directorio donde se encuentran los archivos .mat
input_dir = fullfile('BBDD', 'MrOS', 'f_c4', 'train_c4_MrOS');
% Rutas alternativas:
% input_dir = fullfile('BBDD', 'MrOS', 'f_c4', 'test_c4_MrOS');
% input_dir = fullfile('BBDD', 'MrOS', 'f_c4', 'validation_c4_MrOS');

archivos = dir(fullfile(input_dir, '*.mat'));
num_archivos = length(archivos);
duracion = zeros(num_archivos, 1);

% Selección de archivo(s) a procesar: 1:num_archivos para todos, o un índice fijo
indices_a_procesar = 1; 

%% 2. PROCESAMIENTO Y DETECCIÓN DE VENTANA VÁLIDA
for k = indices_a_procesar
    fprintf('Procesando [%d/%d]: %s\n', k, num_archivos, archivos(k).name);
    
    % Carga de datos
    datos = load(fullfile(input_dir, archivos(k).name));
    fs = double(datos.fs);
    senal = double(datos.EEG_resampled);
    tiempo_horas = length(senal) / (fs * 3600);
    
    % Criterio de duración
    if tiempo_horas < 12
        % Señales de duración normal: se conservan íntegras
        duracion(k) = tiempo_horas;
        hora_inicio = 0;
        hora_fin = tiempo_horas;
    else
        % Señales largas (>12 h): detección de artefactos y desconexión
        ventana_seg = 15 * 60; % Ventana de análisis de 15 minutos
        muestras_ventana = ventana_seg * fs;
        
        % Segmento de referencia basal (primeras 8 horas)
        horas_ref = 8;
        muestras_ref = min(length(senal), round(horas_ref * 3600 * fs));
        num_bloques_ref = floor(muestras_ref / muestras_ventana);
        
        rms_por_bloque = zeros(num_bloques_ref, 1);
        for j = 1:num_bloques_ref
            idx_ref = (j - 1) * muestras_ventana + 1 : j * muestras_ventana;
            rms_por_bloque(j) = rms(senal(idx_ref));
        end
        
        % Cálculo de umbrales basados en IQR sobre la referencia
        Q1 = prctile(rms_por_bloque, 25);
        Q3 = prctile(rms_por_bloque, 75);
        IQR_val = Q3 - Q1;
        umbral_superior = Q3 + 1.5 * IQR_val;
        umbral_inferior = max(0, Q1 - 1.5 * IQR_val);
        
        % Evaluación de calidad por bloque en toda la señal
        num_total_bloques = floor(length(senal) / muestras_ventana);
        es_bueno = zeros(num_total_bloques, 1);
        
        for i = 1:num_total_bloques
            idx = (i - 1) * muestras_ventana + 1 : i * muestras_ventana;
            seg = senal(idx);
            rms_act = rms(seg);
            
            if (rms_act < umbral_superior) && (rms_act > umbral_inferior)
                es_bueno(i) = 1;
            else
                es_bueno(i) = 0;
            end
        end
        
        % La referencia basal se asume válida
        es_bueno(1:num_bloques_ref) = 1;
        
        % Identificación de la componente conexa inicial continua
        bloques_conectados = bwlabel(es_bueno);
        id_region_ref = bloques_conectados(1);
        ultimo_bloque_valido = find(bloques_conectados == id_region_ref, 1, 'last');
        
        hora_inicio = 0;
        hora_fin = (ultimo_bloque_valido * muestras_ventana) / (fs * 3600);
        duracion(k) = hora_fin;
    end
end

%% 3. GENERACIÓN DE GRÁFICAS
% Formato del nombre para visualización
nombre_archivo = strrep(archivos(k).name, '.mat', '');
nombre_titulo = strrep(nombre_archivo, '_', '\_');
T = (0:length(senal) - 1) / (fs * 3600); % Eje temporal en horas

% Figura 1: Señal bruta completa
figure(1);
clf;
plot(T, senal);
title(['Señal EEG Original: ', nombre_titulo], 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Tiempo (Horas)', 'FontSize', 14);
ylabel('Amplitud (\muV)', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

% Figura 2: Señal con límites de recorte detectados
figure(2);
clf;
plot(T, senal, 'Color', [0.5 0.5 0.5]); % Señal en gris
hold on;
xline(hora_inicio, '-r', 'Inicio', 'LineWidth', 2.5, 'FontSize', 14, 'LabelVerticalAlignment', 'top');
xline(hora_fin, '-r', 'Fin', 'LineWidth', 2.5, 'FontSize', 14, 'LabelVerticalAlignment', 'top');
title(['Detección de Ventana Válida: ', nombre_titulo], 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Tiempo (Horas)', 'FontSize', 14);
ylabel('Amplitud (\muV)', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);
hold off;
