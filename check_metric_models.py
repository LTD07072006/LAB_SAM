from huggingface_hub import model_info

names = [
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
]
for name in names:
    try:
        info = model_info(name)
        print(name, "OK", info.pipeline_tag)
    except Exception as exc:
        print(name, "NO", type(exc).__name__, str(exc)[:160])
