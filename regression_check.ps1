param(
    [string]$PythonPath = ".\\.venv\\Scripts\\python.exe",
    [string]$DataFile = "hemograma_data.csv",
    [string]$LegacyOutputDir = "outputs_smoke_legacy",
    [string]$CliOutputDir = "outputs_smoke_cli",
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

if (-not (Test-Path $PythonPath)) {
    throw "Python no encontrado en: $PythonPath"
}

function Invoke-Step {
    param(
        [string]$Title,
        [scriptblock]$Action
    )

    Write-Host "`n=== $Title ===" -ForegroundColor Cyan
    & $Action
}

if (-not $ValidateOnly) {
    Invoke-Step -Title "Generando dataset sintetico" -Action {
        & $PythonPath generate_hemograma_data.py --output $DataFile
        if ($LASTEXITCODE -ne 0) {
            throw "Fallo en generacion de dataset"
        }
    }

    Invoke-Step -Title "Entrenamiento completo via wrappers legacy" -Action {
        & $PythonPath train_and_evaluate.py --data $DataFile --output-dir $LegacyOutputDir --tune --cv-folds 5
        if ($LASTEXITCODE -ne 0) {
            throw "Fallo en entrenamiento legacy"
        }
    }

    Invoke-Step -Title "Entrenamiento completo via CLI modular" -Action {
        & $PythonPath main.py train --data $DataFile --output-dir $CliOutputDir --tune --cv-folds 5
        if ($LASTEXITCODE -ne 0) {
            throw "Fallo en entrenamiento CLI"
        }
    }
}

$tmpValidationScript = Join-Path $scriptRoot "tmp_validate_classes.py"

$validationCode = @"
import pandas as pd
from pathlib import Path

DISEASES = [
    "Sano",
    "Anemia Ferropenica",
    "Anemia Megaloblastica",
    "Infeccion Bacteriana",
    "Infeccion Viral",
    "Trombocitopenia",
    "Policitemia Vera",
    "Leucemia",
    "Alergias",
    "Deshidratacion",
]

roots = ["$LegacyOutputDir", "$CliOutputDir"]
models = ["random_forest", "random_forest_esp32", "xgboost"]

ok = True

ds = pd.read_csv("$DataFile")
present = sorted(ds["condition_name"].unique().tolist())
missing_dataset = [d for d in DISEASES if d not in present]

for root in roots:
    print(f"=== {root} ===")
    if missing_dataset:
        print(f"dataset missing classes: {missing_dataset}")
        ok = False
    else:
        print("dataset classes ok")

    for model_name in models:
        report_path = Path(root) / model_name / "classification_report.txt"
        if not report_path.exists():
            print(f"{model_name}: report missing")
            ok = False
            continue

        report_text = report_path.read_text(encoding="utf-8")
        missing_diseases = [d for d in DISEASES if d not in report_text]
        if missing_diseases:
            print(f"{model_name}: missing diseases in report: {missing_diseases}")
            ok = False
        else:
            print(f"{model_name}: all 10 diseases present in report")

print("STATUS: PASS" if ok else "STATUS: FAIL")
raise SystemExit(0 if ok else 1)
"@

Invoke-Step -Title "Validando cobertura de las 10 enfermedades" -Action {
    Set-Content -Path $tmpValidationScript -Value $validationCode -Encoding UTF8
    try {
        & $PythonPath $tmpValidationScript
        if ($LASTEXITCODE -ne 0) {
            throw "Validacion de clases fallo"
        }
    }
    finally {
        if (Test-Path $tmpValidationScript) {
            Remove-Item $tmpValidationScript -Force
        }
    }
}

Write-Host "`nRegresion completa finalizada correctamente." -ForegroundColor Green
