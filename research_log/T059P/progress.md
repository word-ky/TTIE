# T059-P
Authorization: 89f0baabb74530724c9c78e6014f2d42e15cbcab.
Reuse exact T014 FrozenCLIP.image_embeddings output including all five fixed views; flatten is lossless, with no view selection or averaging.
Use 1 state (5 views) per forward, FP32 GPU 1; direct squared Euclidean FP64 chunks of 4.
No scalar targets opened during provenance/feature/map stage.
Observed environment: bare python missing; project venv works. NVML driver mismatch; torch CUDA available, A6000 free 50.6 GB.
T059-O review transcription correction: actual fit 0.04802930727601051, selector 0.1556149274110794. Selector reused, not fresh.
