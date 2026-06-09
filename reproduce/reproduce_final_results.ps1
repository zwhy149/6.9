$ErrorActionPreference = "Stop"

$py = "C:\Users\wmy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (-not (Test-Path -LiteralPath $py)) {
    throw "Bundled Codex Python was not found: $py"
}

$env:GRID = "0.55,0.65,0.75,0.85,0.92,0.96,0.985"
$env:OBJECTIVE = "accuracy_only"
$env:INCLUDE_GLOBAL_HGB = "0"

& $py ".\code\final_pipeline\rescompact_multisource_ensemble_eval.py"

$env:PRED_PATH = "outputs\rescompact_multisource_3src_ensemble_accuracy_only_predictions.csv"
$env:OUTPUT_PREFIX = "best3src_model"
& $py ".\code\final_pipeline\error_upper_bound_analysis.py"

& $py ".\code\final_pipeline\rescompact_multisource_gate_eval.py"
& $py ".\code\final_pipeline\write_refined_report_and_plots.py"

Write-Host "Final strict report: outputs\refined_strict_result_report.md"
