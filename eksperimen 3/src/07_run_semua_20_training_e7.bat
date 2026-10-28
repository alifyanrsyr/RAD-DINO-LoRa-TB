@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo E7 - Frozen RAW/CLAHE and LoRA RAW/CLAHE, 5 folds each.
echo Fixed E6 development folds; no test image reads.

for %%M in (frozen lora) do (
    for %%V in (raw clahe) do (
        for %%F in (1 2 3 4 5) do (
            if exist "..\output\training\%%M_%%V_fold_%%F\run_summary.json" (
                echo [SKIP] %%M %%V fold %%F; verify by status script at end.
            ) else (
                if exist "..\output\training\%%M_%%V_fold_%%F\latest_training_state.pt" (
                    echo [RESUME] %%M %%V fold %%F
                    python 07_train_%%M_5fold_e7.py --fold %%F --variant %%V --resume
                ) else (
                    if exist "..\output\training\%%M_%%V_fold_%%F" (
                        echo [STOP] Folder already exists without a resumable checkpoint: %%M %%V fold %%F
                        exit /b 2
                    )
                    echo [START] %%M %%V fold %%F
                    python 07_train_%%M_5fold_e7.py --fold %%F --variant %%V
                )
                if errorlevel 1 (
                    echo [STOP] Training failed: %%M %%V fold %%F
                    exit /b 1
                )
            )
        )
    )
)
python 07_status_training_e7.py --require-complete
if errorlevel 1 exit /b 1
echo ALL E7 20 TRAININGS VERIFIED.
endlocal
