import argparse
import os
from pathlib import Path

import torch
from double_sparse_compression.inference import SparsifiedLinear, DoubleSparseLegacy
from transformers import AutoConfig, AutoModelForCausalLM


def load_legacy_tensor(p: str) -> DoubleSparseLegacy:
    legacy_tensor_a = torch.load(os.path.join(p, 'a'), map_location="cpu")
    legacy_tensor_b = torch.load(os.path.join(p, 'b'), map_location="cpu")

    return DoubleSparseLegacy(
        m=legacy_tensor_a.shape[0],
        n=legacy_tensor_b.shape[1],
        k=legacy_tensor_a.shape[1],
        a=legacy_tensor_a,
        b=legacy_tensor_b
    )


def replace_and_save_quantized_layers(
        model_to_be_quantized,
        legacy_model_path,
        is_legacy,
        current_model=None,
        layer_id: int = -1,
        parent_tensor_name="",
):
    if current_model == None:
        current_model = model_to_be_quantized
    for tensor_name, m in current_model.named_children():
        if tensor_name.isnumeric():
            layer_id = int(tensor_name)

        if isinstance(m, torch.nn.Linear):
            assert m.bias is None
            tensor_path = os.path.join(legacy_model_path, f"{layer_id}", f"{parent_tensor_name}.{tensor_name}")
            if os.path.exists(tensor_path):
                if is_legacy:
                    ds_legacy = load_legacy_tensor(tensor_path)
                    ds_module = SparsifiedLinear.from_legacy(ds_legacy, 'cpu')
                else:
                    ds_module = torch.load(tensor_path, 'cpu')
                setattr(current_model, tensor_name, ds_module)
        else:
            replace_and_save_quantized_layers(
                model_to_be_quantized,
                legacy_model_path,
                is_legacy,
                m,
                layer_id,
                tensor_name,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the unquantized model",
    )
    parser.add_argument(
        "--tensors_path",
        type=str,
        required=True,
        help="path to legacy model",
    )
    parser.add_argument(
        "--sparse_strategy",
        type=str,
        default="csr",
        choices=["csr"],
        help="Sparse strategy storage. Options: csr, ptcsr, auto.\nCSR - Compressed Sparse Rows\nPTCSR - Alternative storage format\noptimize_latency - Use the current GPU to determine the optimal storage format to reduce kernel latency",
    )
    parser.add_argument(
        "--tensor_type",
        required=True,
        choices=["legacy", "compressed"],
        help="path to legacy model",
    )
    parser.add_argument(
        "--save_per_layer",
        type=str,
        required=False,
        help="Save the converted quantized model per layer here - useful for benchmarking individual layers",
    )
    parser.add_argument(
        "--torch_pt_path",
        type=str,
        required=False,
        help="Save the converted quantized model per layer here - useful for benchmarking individual layers",
    )

    args, leftovers = parser.parse_known_args()

    # For example, experiment0.8
    tensors_path = Path(args.tensors_path).parts[-1]

    is_legacy = args.tensor_type == 'legacy'

    # For example, outputs/experiment0.8
    # output_path = os.path.join(args.save_per_layer, legacy_model_name)

    if args.save_per_layer is not None:
        for p in os.listdir(args.tensors_path):
            if not os.path.isdir(os.path.join(args.tensors_path, p)): continue

            # Now p is one of 0, 1, ...

            layer_path = os.path.join(args.tensors_path, p)

            for tensor_path in os.listdir(layer_path):
                sublayer_path = os.path.join(layer_path, tensor_path)
                if not os.path.isdir(sublayer_path):
                    continue

                ds_legacy = load_legacy_tensor(sublayer_path)

                sparsified_linear = SparsifiedLinear.from_legacy(ds_legacy, 'cpu')
                output_tensor_path = os.path.join(args.save_per_layer, legacy_model_name, p, tensor_path)

                os.makedirs(Path(output_tensor_path).parent, exist_ok=True)

                print(f'A density = {(sparsified_linear.a_col_vals.shape[0] / (sparsified_linear.m * sparsified_linear.k)) * 100}%')
                print(f'B density = {(sparsified_linear.b_col_vals.shape[0] / (sparsified_linear.m * sparsified_linear.k)) * 100}%')

                torch.save(sparsified_linear, str(output_tensor_path))

    config = AutoConfig.from_pretrained(args.base_model, return_dict=True)

    config.max_position_embeddings = 4096
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.base_model, trust_remote_code=True, torch_dtype=torch.half, config=config
    )


    if args.torch_pt_path:
        not_quantized_weights_path = os.path.join(args.tensors_path, "not_quantized_weights.pt")
        not_quantized_weights = torch.load(not_quantized_weights_path)
        for w in not_quantized_weights.values():
            w.requires_grad = False
        model.load_state_dict(not_quantized_weights, strict=False)

        replace_and_save_quantized_layers(
            model,
            args.tensors_path,
            is_legacy
        )
        torch.save(model, args.torch_pt_path)

