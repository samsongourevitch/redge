import time
import torch

from samplers.ddim import redge


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 64
    num_classes = 100
    n_steps = 5
    n_calls = 5000
    n_seeds = 20
    t_1 = 0.2

    total_times = []
    per_call_times = []
    per_step_times = []

    with torch.no_grad():
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            logits = torch.randn(batch_size, num_classes, device=device)

            # Warmup to avoid one-time overheads in the timing.
            _ = redge(logits, n_steps=n_steps, t_1=t_1)
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(n_calls):
                _ = redge(logits, n_steps=n_steps, t_1=t_1)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            total_s = end - start
            total_times.append(total_s)
            per_call_times.append(total_s / n_calls)
            per_step_times.append(total_s / (n_calls * n_steps))

    total_times_t = torch.tensor(total_times)
    per_call_times_t = torch.tensor(per_call_times)
    per_step_times_t = torch.tensor(per_step_times)

    print("redge timing")
    print(f"device: {device}")
    print(f"batch_size: {batch_size}, num_classes: {num_classes}")
    print(f"n_steps: {n_steps}, n_calls: {n_calls}, n_seeds: {n_seeds}")
    print(
        f"total: {total_times_t.mean():.4f}s ± {total_times_t.std(unbiased=False):.4f}s"
    )
    print(
        f"per call: {per_call_times_t.mean() * 1e3:.4f} ms ± {per_call_times_t.std(unbiased=False) * 1e3:.4f} ms"
    )
    print(
        f"per step: {per_step_times_t.mean() * 1e6:.4f} us ± {per_step_times_t.std(unbiased=False) * 1e6:.4f} us"
    )


if __name__ == "__main__":
    main()
