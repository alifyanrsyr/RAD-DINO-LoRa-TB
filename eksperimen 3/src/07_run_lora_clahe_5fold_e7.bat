@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo EXPERIMENT 7: LORA CLAHE, FOLD 1-5
for %%F in (1 2 3 4 5) do (
    if exist "..\output\training\lora_clahe_fold_%%F\run_summary.json" (
        echo [SKIP] Fold %%F has run_summary.json; verify via status script.
    ) else (
        if exist "..\output\training\lora_clahe_fold_%%F\latest_training_state.pt" (
            echo [RESUME] Fold %%F
            python 07_train_lora_5fold_e7.py --fold %%F --resume
        ) else (
            if exist "..\output\training\lora_clahe_fold_%%F" (
                echo [ERROR] Fold %%F folder exists but has no valid resume state.
                exit /b 2
            )
            echo [START] Fold %%F
            python 07_train_lora_5fold_e7.py --fold %%F
        )
        if errorlevel 1 (
            echo [ERROR] Training stopped on fold %%F
            exit /b 1
        )
    )
)
python 07_status_training_e7.py --require-complete
if errorlevel 1 exit /b 1
echo Training batch completed. Confirm COMPLETED 5/5 above.
endlocal
