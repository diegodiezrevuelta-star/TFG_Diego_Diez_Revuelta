% =========================================================================
% Preprocesamiento de los registros de EEG 
%
% Este script es utilizado para con todas las bases de datos del proyecto.
% Hay que tener en cuenta que algunas bases de datos presentan las subdivisiones (train, test y validation) guardadas en carpetas separadas.
% Se deberá ejecutar este  script para cada una de esas subcarpetas de forma individual, modificando 
% la variable 'input_dir' en cada caso.
% =========================================================================

% --- CONFIGURACIÓN DE RUTA ---
% Cambia esta ruta a la carpeta donde se encuentren tus archivos .mat
input_dir = './BBDD/APPLES/apple_procesado'; 

% Buscamos todos los archivos .mat que haya dentro de esa carpeta
archivos = dir(fullfile(input_dir, '*.mat'));
num_archivos = length(archivos);

% --- PARÁMETROS DE SEÑAL Y FILTRO ---
limite_horas = 12; 
fs = 100; 
puntos_limite = limite_horas * 3600 * fs; % 4.320.000 puntos

% Diseño del filtro Butterworth 0.5 - 30 Hz (Fase cero)
lowcut = 0.5;
highcut = 30.0;
nyq = fs / 2;
[b, a] = butter(4, [lowcut, highcut] / nyq, 'bandpass'); % 4º orden

% --- BUCLE DE PROCESAMIENTO ---
for i = 1:num_archivos
    nombre_archivo = archivos(i).name;
    ruta_completa = fullfile(input_dir, nombre_archivo);
    
    fprintf('Procesando, Filtrando y Sobrescribiendo (%d/%d): %s\n', i, num_archivos, nombre_archivo);
    
    % 1. CARGA SELECTIVA (Solo cargamos 'senal_final' para no saturar la memoria) % Nuestros datos presentan la señal guardada como senal_final junto con otra información de interés.
    try
        m_in = load(ruta_completa, 'senal_final');
        if ~isfield(m_in, 'senal_final')
            fprintf('Aviso: El archivo %s no contiene la variable "senal_final". Saltando.\n', nombre_archivo);
            continue;
        end
        senal = single(m_in.senal_final); 
        
    catch
        fprintf('Error al abrir el archivo %s. Saltando.\n', nombre_archivo);
        continue;
    end
    
    % 2. TRUNCADO A 12 HORAS
    puntos_actuales = length(senal);
    if puntos_actuales > puntos_limite
        real_len = puntos_limite;
    else
        real_len = puntos_actuales;
    end
    
    % Extraer señal real útil 
    senal_real = double(senal(1:real_len));
    
    % 3. FILTRADO  (0.5 - 30 Hz)
    senal_filtrada = filtfilt(b, a, senal_real);
    
    % 4. NORMALIZACIÓN Z-SCORE 
    mu = mean(senal_filtrada);
    sigma = std(senal_filtrada);
    if sigma > 1e-6
        senal_norm = (senal_filtrada - mu) / (sigma + 1e-6);
    else
        senal_norm = senal_filtrada;
    end
    
    % 5. PADDING DE ZEROS
    senal_final = zeros(1, puntos_limite, 'single');
    senal_final(1:real_len) = single(senal_norm); % Usamos single para ahorrar memoria

    % 6. GUARDAMOS
    % El parámetro '-append' asegura que el resto de variables/etiquetas del archivo queden INTACTAS
    save(ruta_completa, 'senal_final', '-append');
end

disp('¡Proceso completado!');
