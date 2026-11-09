# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-11-09

### Added
- **Early Stopping Support**: Added early stopping mechanism based on validation `COCOAP50_95`:
  - `yolox/exp/yolox_base.py`: Added `early_stopping_patience` and `early_stopping_min_delta` configuration attributes.
  - `yolox/core/trainer.py`: Implemented validation checkpoint tracking, counter persistence across resume (`no_improvement_epochs`), and early exit logic when mAP fails to improve.
  - `tools/train.py`: Added CLI flags `--early-stopping-patience` and `--early-stopping-min-delta`.
  - Configured in `exps/default/yolox_s_train.py` (patience = 15) and `exps/default/yolox_s_finetune.py` (patience = 8).

## [0.1.0] - 2026-09-15

### Fixed
- **COCO Evaluation on Windows**: Fixed `subprocess.CalledProcessError: Command '['where', 'cl']'` crash in `yolox/evaluators/coco_evaluator.py`. On Windows systems without MSVC C++ build tools, evaluation now automatically falls back cleanly to standard `pycocotools.cocoeval.COCOeval` instead of attempting C++ JIT compilation with `FastCOCOEvalOp`.
- **PyTorch 2.6+ `torch.load` Compatibility**: Added `weights_only=False` to `torch.load` calls in `tools/eval.py`, `yolox/core/trainer.py`, and `tools/export_onnx.py` to prevent `pickle.UnpicklingError` caused by default `weights_only=True` in PyTorch 2.6+.
