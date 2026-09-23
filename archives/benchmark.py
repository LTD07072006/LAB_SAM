"""Latency benchmark with mean, p50 and p95 reporting."""
import argparse, statistics, time
from pathlib import Path
import numpy as np
from config import DEFAULT_CONFIG
from fire_detector import FireDetector
from locator import GridMap, intersect_ray_with_grid_result
from main_integrated import make_default_camera
from localization import localize_pixels, localize_with_uncertainty

def percentile(values, q): return float(np.percentile(values, q)) if values else float("nan")

def benchmark(samples, model, repeats=20, warmup=3, device=None, mode="single", uncertainty_samples=64):
    detector = FireDetector(model, device=device, threshold=DEFAULT_CONFIG.confidence_threshold); detector.warmup(warmup)
    camera, _ = make_default_camera(); grid = GridMap(); paths = sorted(p for p in samples.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    ai_times, geo_times, total_times = [], [], []; detected = hits = 0
    for path in paths:
        local_ai, local_geo = [], []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter(); result = detector.detect(path); t1 = time.perf_counter()
            if not result.detected: continue
            detected += 1
            t2 = time.perf_counter()
            if mode == "single":
                C, ray = camera.pixel_to_ray(*result.pixel)
                hit = intersect_ray_with_grid_result(C, ray, grid, DEFAULT_CONFIG.ray_max_distance, DEFAULT_CONFIG.ray_coarse_step, DEFAULT_CONFIG.ray_bisection_iterations)
            elif mode == "multi":
                px = np.asarray(result.pixel, dtype=np.float64); span = max(2.0, min(result.size) * 0.015)
                pixels = np.column_stack([np.linspace(px[0] - span, px[0] + span, DEFAULT_CONFIG.multi_ray_columns), np.full(DEFAULT_CONFIG.multi_ray_columns, px[1])])
                hit = localize_pixels(camera, grid, pixels, max_dist=DEFAULT_CONFIG.ray_max_distance, step=DEFAULT_CONFIG.ray_coarse_step, bisection_iterations=DEFAULT_CONFIG.ray_bisection_iterations)
            elif mode == "uncertainty":
                hit = localize_with_uncertainty(camera, grid, result.pixel, pixel_sigma=DEFAULT_CONFIG.uncertainty_pixel_sigma, samples=uncertainty_samples, max_dist=DEFAULT_CONFIG.ray_max_distance, step=DEFAULT_CONFIG.ray_coarse_step, bisection_iterations=DEFAULT_CONFIG.ray_bisection_iterations)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            t3 = time.perf_counter()
            ai_ms, geo_ms, total_ms = (t1-t0)*1000, (t3-t2)*1000, (t3-t0)*1000
            local_ai.append(ai_ms); local_geo.append(geo_ms); ai_times.append(ai_ms); geo_times.append(geo_ms); total_times.append(total_ms); hits += int(hit.hit)
        print(f"{path.name:16} ai={statistics.fmean(local_ai) if local_ai else float('nan'):.2f}ms geo={statistics.fmean(local_geo) if local_geo else float('nan'):.2f}ms")
    print(f"mode={mode} samples={len(paths)} detected_runs={detected} hit_runs={hits}")
    for label, values in (("ai", ai_times), ("geo", geo_times), ("total", total_times)):
        p50, p95 = percentile(values, 50), percentile(values, 95); fps = 1000/p95 if values and p95 > 0 else float('nan')
        print(f"{label:5} mean={statistics.fmean(values) if values else float('nan'):.2f}ms p50={p50:.2f}ms p95={p95:.2f}ms fps@p95={fps:.2f}")

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--samples", type=Path, default=DEFAULT_CONFIG.sample_dir); parser.add_argument("--model", type=Path, default=DEFAULT_CONFIG.model_path); parser.add_argument("--repeats", type=int, default=20); parser.add_argument("--warmup", type=int, default=3); parser.add_argument("--device", default=None); parser.add_argument("--mode", choices=("single", "multi", "uncertainty"), default="single"); parser.add_argument("--uncertainty-samples", type=int, default=64); args = parser.parse_args(); benchmark(args.samples, args.model, args.repeats, args.warmup, args.device, args.mode, args.uncertainty_samples)

if __name__ == "__main__": main()
