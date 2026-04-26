# Clasificacion Multiclase de Hemograma para ESP32

Proyecto para generar datos sinteticos de hemograma (50,000 registros), entrenar clasificadores multiclase y preparar despliegue en ESP32 con foco en sensibilidad para clases criticas.

## Arquitectura actual

- `hemodiagnostico/`: paquete principal del dominio.
  - `config.py`: carga y fusion de configuracion del proyecto.
  - `data_generation.py`: logica medica de simulacion de dataset.
  - `model_training.py`: tuning, evaluacion, CV y exportacion ESP32.
  - `cli.py`: CLI unificada (`generate`, `train`, `full`).
- `configs/project_config.json`: configuracion central editable.
- `main.py`: entrada unica recomendada.
- `generate_hemograma_data.py`: wrapper compatible al generador.
- `train_and_evaluate.py`: wrapper compatible al entrenamiento.

## Flujo recomendado

1. Generar dataset:

```powershell
python main.py generate
```

2. Entrenar y evaluar:

```powershell
python main.py train --tune --cv-folds 5 --output-dir outputs_step5
```

3. Flujo completo en un comando:

```powershell
python main.py full
```

## Validacion rapida antes de cambios

Regresion end-to-end (genera datos, entrena por ambas rutas y valida clases):

```powershell
powershell -ExecutionPolicy Bypass -File .\regression_check.ps1
```

Solo validacion de reportes existentes:

```powershell
powershell -ExecutionPolicy Bypass -File .\regression_check.ps1 -ValidateOnly
```

Pruebas unitarias/smoke:

```powershell
python -m pytest -q
```

## Configuracion central

Ajusta parametros en `configs/project_config.json`:

- `data_generation.output_csv`
- `data_generation.random_seed`
- `training.data_path`
- `training.output_dir`
- `training.tune`
- `training.cv_folds`
- `training.random_seed`

## Artefactos principales

En la carpeta de salida (`training.output_dir`) se generan:

- `metrics_summary.csv`
- `cv_summary.csv` (si `cv_folds >= 2`)
- `random_forest/` (matriz + reporte)
- `random_forest_esp32/` (matriz + reporte)
- `xgboost/` (si disponible)
- `esp32_export/`
  - `esp32_rf_model.h`
  - `esp32_preprocess.h`
  - `feature_manifest.csv`
  - `class_labels.csv`
  - `deployment_summary.json`
  - `esp32_inference_template.ino`

## Despliegue en ESP32

El modelo elegido para firmware es `Random Forest ESP32` (compacto) para balancear memoria y recall clinico alto. Usa siempre:

1. Escalado con `esp32_preprocess.h`.
2. Mismo orden de features definido en `feature_manifest.csv`.
3. Mapeo de salida con `class_labels.csv`.
