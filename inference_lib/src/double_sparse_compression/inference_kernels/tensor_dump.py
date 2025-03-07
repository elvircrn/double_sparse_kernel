import os

import torch
from double_sparse_compression import SparsifiedLinear

if __name__ == '__main__':
    input_path = '/mnt/6e3c126c-c6bb-43eb-9d82-1e59b2111688/ecrncevi/double_sparse_data/compressed_csr/experiment0.70'
    output_path = '/mnt/6e3c126c-c6bb-43eb-9d82-1e59b2111688/ecrncevi/double_sparse_data/compressed_csr/bin/experiment0.70'

    for layer_id in os.listdir(input_path):
        folder = os.path.join(input_path, layer_id)
        out_folder = os.path.join(output_path, layer_id)
        if not os.path.isdir(folder):
            continue
        os.makedirs(out_folder, exist_ok=True)
        for tensor_name in os.listdir(folder):
            in_tensor_path = os.path.join(folder, tensor_name)
            out_tensor_path = os.path.join(out_folder, tensor_name)
            os.makedirs(out_tensor_path, exist_ok=True)
            doublesparse_module: SparsifiedLinear = torch.load(in_tensor_path)

            with open(os.path.join(out_tensor_path, "a_row_offsets.bin"), "wb") as out_file:
                out_file.write(doublesparse_module.a_row_offsets.numpy().tobytes())

            with open(os.path.join(out_tensor_path, "a_col_vals.bin"), "wb") as out_file:
                out_file.write(doublesparse_module.a_col_vals.numpy().tobytes())

            with open(os.path.join(out_tensor_path, "b_row_offsets.bin"), "wb") as out_file:
                out_file.write(doublesparse_module.b_row_offsets.numpy().tobytes())

            with open(os.path.join(out_tensor_path, "b_col_vals.bin"), "wb") as out_file:
                out_file.write(doublesparse_module.b_col_vals.numpy().tobytes())

            with open(os.path.join(out_tensor_path, "meta.txt"), "w") as out_file:
                out_file.write(f"{doublesparse_module.m} {doublesparse_module.n} {doublesparse_module.k} {doublesparse_module.non_zero_rows}")
