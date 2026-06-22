import time

import torch
from PIL import Image
from rfdetr import RFDETRMedium, RFDETRNano

IMAGE_PATH = "000000039769.jpg"
THRESHOLD = 0.5
WARMUP = 10
RUNS = 50

use_cuda = torch.cuda.is_available()
image = Image.open(IMAGE_PATH).convert("RGB")


def sync():
    if use_cuda:
        torch.cuda.synchronize()


def time_loop(fn, runs):
    sync()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    sync()
    return (time.perf_counter() - start) / runs * 1000.0


def benchmark(model_cls):
    model = model_cls()
    model.optimize_for_inference(dtype=torch.float16)

    res = model.model.resolution
    net = model.model.inference_model
    dummy = torch.randn(
        1,
        model.model_config.num_channels,
        res,
        res,
        device=model.model.device,
        dtype=model._optimized_dtype,
    )

    # Pure GPU forward pass: isolates model compute, where input resolution dominates.
    for _ in range(WARMUP):
        net(dummy)
    forward_ms = time_loop(lambda: net(dummy), RUNS)

    # End-to-end predict(): adds fixed Python pre/post-processing overhead per call.
    for _ in range(3):
        model.predict(image, threshold=THRESHOLD)
    e2e_ms = time_loop(lambda: model.predict(image, threshold=THRESHOLD), RUNS)

    del model
    if use_cuda:
        torch.cuda.empty_cache()

    return res, forward_ms, e2e_ms


print(f"Device: {'CUDA' if use_cuda else 'CPU'} | precision=FP16 | warmup={WARMUP}, runs={RUNS}\n")
print(f"{'Model':7s} | {'Input res':9s} | {'Forward (GPU)':>14s} | {'End-to-end':>12s}")
print("-" * 54)

forward = {}
for model_cls, name in [(RFDETRMedium, "Medium"), (RFDETRNano, "Nano")]:
    res, forward_ms, e2e_ms = benchmark(model_cls)
    forward[name] = forward_ms
    print(f"{name:7s} | {res:>4d}x{res:<4d} | {forward_ms:9.2f} ms | {e2e_ms:7.1f} ms")

print(
    f"\nForward pass: Nano is {forward['Medium'] / forward['Nano']:.2f}x faster than Medium "
    f"({forward['Medium']:.2f} ms -> {forward['Nano']:.2f} ms)"
)
print("Reference (official, T4 + TensorRT FP16, bs=1): Nano 384x384 @ 2.3 ms, Medium 576x576 @ 4.4 ms")
