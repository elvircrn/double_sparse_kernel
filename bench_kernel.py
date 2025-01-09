import argparse
import os
import time

import numpy as np
import torch
from scipy.stats import gmean
import io
import pandas as pd

from double_sparse_compression import SparsifiedLinear
from double_sparse_compression.inference import FeatureFlags
from double_sparse_compression.inference_kernels.kernel_selector import get_doublesparse_mul_timer

cutlass_str = """m,n,k,Runtime
11008,11008,1,1.04251
11008,4096,1,0.395374
4096,11008,1,0.385591
4096,4096,1,0.148815
"""

cutlass_data = io.StringIO(cutlass_str)
cutlass_runs = pd.read_csv(cutlass_data)


def doublesparse_mul_timer(doublesparse_device: SparsifiedLinear, x, feature_flag: FeatureFlags, batch_size):
    result = torch.empty(1).cpu().float()
    y = torch.zeros(doublesparse_device.m * batch_size, dtype=x.dtype, device=x.device).flatten().contiguous()

    get_doublesparse_mul_timer()(
        doublesparse_device.m,
        doublesparse_device.n,
        doublesparse_device.k,
        doublesparse_device.a_row_offsets,
        doublesparse_device.a_col_vals,
        doublesparse_device.b_row_offsets,
        doublesparse_device.b_col_vals,
        doublesparse_device.non_zero_rows,
        batch_size,
        x,
        y,
        result,
        feature_flag
    )

    return y, result.item()


if __name__ == "__main__":
    torch_runs = {}

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--tensor_path",
        type=str,
        required=True,
        help="Path to folder containing the tensors of the form"
             "model_path/"
             "   0/"
             "       tensor0"
             "       tensor1",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to results *.csv file.",
    )

    args = parser.parse_args()

    with open(args.output_path, "w") as f:
        base_path = args.tensor_path

        seed = 1
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        device = torch.device("cuda")

        for m in [4096, 11008]:
            for n in [4096, 11008]:
                cutlass_run = cutlass_runs[
                    (cutlass_runs["m"] == m) &
                    (cutlass_runs["n"] == n) &
                    (cutlass_runs["k"] == 1)]["Runtime"].item()
                torch_runs[(m, n)] = cutlass_run

        csr_folders = os.listdir(base_path)

        csr_folders.sort()

        methods = [
            FeatureFlags.CSR,
        ]

        f.write("Layer;Tensor Name;M;N;K")

        for method in [FeatureFlags.CSC] + methods:
            f.write(f";{method.pretty()} CSR (ms)")

        f.write("\n")

        benchmark_results_ms = []
        benchmark_speed_up = []

        def generate_x_fp32(n, upper_bound=3):
            x_fp32 = ((torch.rand(n) - 0.5) * 4 * upper_bound).int()
            return x_fp32.float()


        x_fp32 = generate_x_fp32(n)
        x_fp16_device = x_fp32.cuda(device=device).half()

        for layer_id in csr_folders:
            folder = os.path.join(base_path, layer_id)

            folders_modified_csr = os.path.join(base_path, layer_id)
            if not os.path.isdir(folder):
                continue

            for p in os.listdir(folder):
                # if 'up_proj' not in p: continue
                tensor_path = os.path.join(folder, p)
                doublesparse_module = torch.load(tensor_path)

                m = doublesparse_module.m
                n = doublesparse_module.n
                k = doublesparse_module.k
                batch_size = 1


                doublesparse_module_device = doublesparse_module.to(device=device)

                dense_speed_up = 0
                baseline_speed_up = 0

                torch_run = torch_runs[(doublesparse_module_device.m, doublesparse_module_device.n)]

                print(f"Running {m} x {n} x {k} Batch Size = {batch_size} Densities {doublesparse_module_device.a_row_offsets[-1] / (m * k):.2f} {doublesparse_module_device.b_row_offsets[-1] / (k * n):.2f}")
                f.write(f"{layer_id};{p};{m};{n};{k};{torch_run:.4f}")

                for flag in methods:
                    print(f"Running {repr(flag)} on {layer_id}.{p}")

                    y_csr, this_algorithm = doublesparse_mul_timer(doublesparse_module_device, x_fp16_device, flag, batch_size)

                    speed_up = torch_run / this_algorithm

                    print(
                        f"\t{repr(flag)} running {this_algorithm} ms {speed_up:.2f}X speed-up vs torch {torch_run} ms"
                    )

                    baseline_speed_up = speed_up
                    f.write(f";{this_algorithm:.4f}")

                    benchmark_results_ms.append(this_algorithm)
                    benchmark_speed_up.append(baseline_speed_up)

                f.write("\n")
                f.flush()
                print("\n\n")

            print(f"Total benchmark geomean = {gmean(benchmark_results_ms)}")
            print(f"Total benchmark speed-up geomean = {gmean(benchmark_speed_up)}")

            print(f"Total benchmark mean = {np.array(benchmark_results_ms).mean()}")
            print(f"Total benchmark speed-up mean= {np.array(benchmark_speed_up).mean()}")

            print("\n\n")
