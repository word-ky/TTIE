# LSRW: PromptIR / PromptIR+DCTTA near-identity outputs — execution check

2026-09-26, after the LSRW reference gate opened (receipt `fe44f42d…`). This check is about execution only. It ran in a scratch directory (`/root/autodl-tmp/TTIE/T075B/scratch/`, scripts in `T075B/code/pir_probe.py` and `pir_geom.py`). No frozen row was replaced or rerun.

**Conclusion: this is how the method behaves on LSRW, not an execution bug.** The frozen five-task PromptIR (`epoch=80.ckpt`) leaves LSRW lows almost unchanged. The same checkpoint brightens SDSD-indoor lows 4 to 5×. The difference comes from image content, not geometry.

## Evidence

1. **Output vs input luminance** (mean of the output clipped to [0,1] vs mean of the canonical low, all images):

   | Images | low | PromptIR | PromptIR+DCTTA | RetinexFormer | SNR-Aware | Ours-TTT |
   |---|---|---|---|---|---|---|
   | Huawei (30, 960×720) | 0.0660 | 0.0672 | 0.0667 | 0.367 | 0.448 | 0.305 |
   | Nikon (20, 960×640) | 0.1685 | 0.1686 | 0.1684 | 0.353 | 0.413 | 0.310 |
   | SDSD-indoor (18 sampled, 960×512) | 0.0954 | 0.4342 | 0.4792 | | | |

2. **Right input, same model.** A scratch rerun of the frozen static code path used `LowOnlyPromptTestDataset` and `load_promptir`, with the checkpoint SHA256 `206baf0d…4d4a` (identical to SDSD). It ran on `Huawei__2037.png` and `Nikon__3003.png`:
   - The input tensor fed to the model equals the staged PNG decoded by cv2, BGR→RGB, /255 (`torch.equal` True). The PNGs are uint8 with 3 channels, and the tensor shapes are (3,720,960) and (3,640,960).
   - Value ranges are [0, 0.859] and [0, 0.784]. Per-channel means are balanced: 0.083/0.075/0.066 and 0.062/0.064/0.064.
   - Both heights are multiples of 16, so the loader's center crop does nothing. The runner asserts that output shape equals input shape, and there is no resize anywhere in the path.
   - Both rerun outputs reproduce the frozen `output_tensor_sha256` exactly.
   - Output − input: mean +0.002 / +0.003, mean absolute 0.004 / 0.005.

3. **Geometry vs content** (scratch, same model; gain = output mean / input mean):

   | Input | Native | Changed geometry |
   |---|---|---|
   | LSRW Huawei__2037 | 1.02–1.03 | center-crop 512×960 / resize 512×960: 1.02 |
   | LSRW Huawei__2050 | 1.02 | center-crop 512×960: 1.03; resize 512×960: 1.02 |
   | LSRW Nikon__3003 | 1.05 | center-crop 512×960: 1.07; resize 512×960: 1.00 |
   | SDSD pair11__0177 | 5.60 | resize 720×960: 5.33; reflect-pad 720×960: 5.66 |
   | SDSD pair4__0188 | 4.03 | resize 720×960: 3.59; reflect-pad 720×960: 4.04 |

   LSRW images at SDSD geometry are still left almost unchanged. SDSD images at LSRW geometry are still enhanced. So the geometry does not cause the behaviour.

PromptIR is a blind all-in-one model: the prompts pick the degradation type implicitly. On LSRW lows it does not activate its low-light enhancement. DCTTA's domain-level adaptation over the 50 LSRW lows does not change that; its output means stay within 0.001 of static. Both rows stay as frozen. This note documents the behaviour for the paper.
