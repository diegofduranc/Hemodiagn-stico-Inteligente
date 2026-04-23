# Clasificacion Multiclase de Hemograma para ESP32

Proyecto para generar datos sinteticos de hemograma (50,000 registros), entrenar modelos multiclase y evaluar metricas con foco en sensibilidad para condiciones criticas (Leucemia y Trombocitopenia).

## Estructura

- `generate_hemograma_data.py`: genera `hemograma_data.csv` con 10 clases clinicas.
- `train_and_evaluate.py`: entrena Random Forest y XGBoost con `RobustScaler`, crea matriz de confusion, classification report, F1 macro y AUC-ROC.
- `requirements.txt`: dependencias.

## Clases modeladas

1. Sano
2. Anemia Ferropenica
3. Anemia Megaloblastica
4. Infeccion Bacteriana
5. Infeccion Viral
6. Trombocitopenia
7. Policitemia Vera
8. Leucemia
9. Alergias
10. Deshidratacion

## Instalacion

```powershell
pip install -r requirements.txt
```

## 1) Generar dataset sintetico

```powershell
python generate_hemograma_data.py --output hemograma_data.csv
```

Salida esperada:
- Archivo `hemograma_data.csv` con 50,000 filas.
- Distribucion de clases impresa en consola.

## 2) Entrenar y evaluar

```powershell
python train_and_evaluate.py --data hemograma_data.csv --output-dir outputs
```

## Resultados generados

En `outputs/`:
- `metrics_summary.csv`
- `random_forest/classification_report.txt`
- `random_forest/confusion_matrix.csv`
- `random_forest/confusion_matrix.png`
- `xgboost/classification_report.txt` (si xgboost esta instalado)
- `xgboost/confusion_matrix.csv` (si xgboost esta instalado)
- `xgboost/confusion_matrix.png` (si xgboost esta instalado)
- `esp32_export/rf_pipeline.joblib`
- `esp32_export/esp32_rf_model.h` (si `micromlgen` esta instalado)

## Enfoque de seguridad clinica

- Se usa `RobustScaler` para mitigar outliers, especialmente de la clase Leucemia.
- Se aplican pesos de clase para aumentar Recall en:
  - `Trombocitopenia` (id 5)
  - `Leucemia` (id 7)
- Se reportan explicitamente:
  - `Recall` por clase critica
  - `F1-Score Macro`
  - `AUC-ROC Macro (OvR)`

## Nota para despliegue en ESP32

Para inferencia embebida, el Random Forest exportado con `micromlgen` es mas adecuado que XGBoost por memoria y complejidad. El archivo de cabecera generado incluye el modelo y el orden exacto de features.
