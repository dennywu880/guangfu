# Final UI Dependency Graph

## Verified delivery chain

```text
TG_UNet_DefectSeg.exe
└── PyInstaller entry: exe_launcher.py
    ├── subprocess: D:\Anaconda\envs\defect_ris\pythonw.exe
    └── gradio_desktop_window.py
        ├── PySide6.QtWebEngineWidgets.QWebEngineView
        └── dynamic import: app.py
            └── inference_engine.DefectInferenceEngine
                ├── dynamic file import: 43_final_testB_gate_v2_and_m2_metrics.py
                │   ├── dataset.RefELDataset
                │   ├── model_text.TextGuidedUNet
                │   └── train_text.FormalTextCache
                │       ├── dataset.RefELDataset
                │       └── model_text
                ├── checkpoints/ (five Final V3 weights)
                ├── runtime_data/presence_gate_v2/presence_gate_v2_result.json
                └── datasets/ metadata for query catalogue and automatic GT
```

## Evidence

- Root and `dist` EXEs have the same SHA256: `F07BC0D6C23716D54670306A9048A7F362CFE6958B2CE8FCA739ADE8B217910C`.
- PyInstaller archive inspection identifies `exe_launcher` as the EXE script entry.
- `TG_UNet_DefectSeg.spec` declares `Analysis(['exe_launcher.py'])` with empty `datas`, `hiddenimports` and `runtime_hooks`.
- `exe_launcher.py` resolves `gradio_desktop_window.py` beside the EXE and launches it through the verified environment's `pythonw.exe`.
- `gradio_desktop_window.py` imports `app` inside its server worker and embeds the local Gradio URL in `QWebEngineView`.
- `app.py` constructs `DefectInferenceEngine`; the engine dynamically loads Step 43 by file path so UI inference shares the frozen Final V3 constants and checkpoint logic.

## Rejected legacy chain

```text
frozen_main.py → desktop_app.py → inference_engine.py
```

This was a separate native-form/onedir prototype. It is not referenced by the current EXE archive or current spec. Its build script also invoked the current launcher-based spec, making the old onedir instructions internally inconsistent. The chain and its staging helper were removed.
