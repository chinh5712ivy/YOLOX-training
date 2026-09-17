# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-09-15

### Fixed
- **COCO Evaluation on Windows**: Fixed `subprocess.CalledProcessError: Command '['where', 'cl']'` crash in `yolox/evaluators/coco_evaluator.py`. On Windows systems without MSVC C++ build tools, evaluation now automatically falls back cleanly to standard `pycocotools.cocoeval.COCOeval` instead of attempting C++ JIT compilation with `FastCOCOEvalOp`.
- **PyTorch 2.6+ `torch.load` Compatibility**: Added `weights_only=False` to `torch.load` calls in `tools/eval.py`, `yolox/core/trainer.py`, and `tools/export_onnx.py` to prevent `pickle.UnpicklingError` caused by default `weights_only=True` in PyTorch 2.6+.
