P=/root/miniconda3/bin/python
T=/root/autodl-tmp/TTIE/T074B
C=$T/code
TC=/root/autodl-tmp/TTIE/T074C
R=$TC/runs
L=$T/targets/SDSD_indoor
OURS_SRC=/root/autodl-tmp/TTIE/T073C/recovery/source
OURS_MANIFEST=/root/autodl-tmp/TTIE/T073C/recovery/artifacts/T073C_execution_manifest.json
B73=/root/autodl-tmp/TTIE/T073B
AZ=/root/autodl-tmp/TTIE/T072AZ
CLEAN="env -u CUBLAS_WORKSPACE_CONFIG -u OMP_NUM_THREADS -u MKL_NUM_THREADS -u OPENBLAS_NUM_THREADS -u MKL_THREADING_LAYER"
AZ_ENV="env CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
QP=/root/autodl-tmp/envs/quadprior/bin/python
QC=/root/autodl-tmp/TTIE/T073E/code
QS=/root/autodl-tmp/TTIE/T073E/source/QuadPrior
WT=/root/autodl-tmp/TTIE/weights
QW="--coco-checkpoint $WT/COCO-final.ckpt --vae-checkpoint $WT/main-epoch=00-step=7000.ckpt --control-checkpoint $WT/control_sd15_ini.ckpt"
QPIN="--expected-coco-sha256 de8d5d17d49b60dd01e48dcbd8cd64ffad818de68d2a3969f774575b47a1c9ab --expected-vae-sha256 43a1143cbf83e4823db6303bf914fca5b7a8f8d25c43ab12554d5ad6d44ade70 --expected-control-sha256 3104871007e81e7b0d599c93392d124081b91da57889fcdda8a3dde70a1c78d9"
MP=/root/autodl-tmp/envs/mri/bin/python
MC=/root/autodl-tmp/TTIE/T073D/code
MS=/root/autodl-tmp/TTIE/T073D/source/MR-Illuminate
SNAP=/root/autodl-tmp/cache/hf/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
MENV="env HF_HOME=/root/autodl-tmp/cache/hf"
MPIN="--expected-vae-sha256 43a1143cbf83e4823db6303bf914fca5b7a8f8d25c43ab12554d5ad6d44ade70 --expected-sd-revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
