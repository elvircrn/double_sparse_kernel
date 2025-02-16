import argparse
import os
from pathlib import Path

import torch
from double_sparse_compression.inference import SparsifiedLinear, DoubleSparseLegacy
from transformers import AutoConfig, AutoModelForCausalLM
import time
from enum import IntEnum
from typing import Tuple

import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer, StaticCache

from modelutils import suspend_nn_inits


torch.autograd.set_grad_enabled(False)

torch.set_printoptions(sci_mode=False)

try:
    import safetensors  # noqa: F401

    has_safetensors = True
except ModuleNotFoundError:
    has_safetensors = False


class Mode(IntEnum):
    DENSE = 0
    QUANTIZED = 1
    TORCH_PT = 2


def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True,
    )[0]
    new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token


class InferenceDemo:
    def __init__(
            self, pretrained_model_path: str, quantized_model_path, flag, device="cuda", torchscript=False, backend=None, model=None
    ):
        self.flag = flag
        self.device = device
        self.dtype = torch.float16
        self.torchscript = torchscript
        self.backend = backend


        if model != None:
            self.config = AutoConfig.from_pretrained(pretrained_model_path, torchscript=self.torchscript)
            self.model = model
        elif flag == Mode.TORCH_PT:
            self.config = AutoConfig.from_pretrained(pretrained_model_path, torchscript=self.torchscript)
            self.model = torch.load(quantized_model_path)
        elif flag == Mode.QUANTIZED:
            with suspend_nn_inits():
                with torch.no_grad():
                    self.config = AutoConfig.from_pretrained(
                        quantized_model_path, torchscript=self.torchscript, return_dict=True, from_tf=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        pretrained_model_name_or_path=quantized_model_path,
                        trust_remote_code=True,
                        config=self.config,
                        from_tf=False,
                        weights_only=False,
                        low_cpu_mem_usage=True,
                        device_map="cpu",
                    )
                    print("Finished loading")
        else:
            with suspend_nn_inits():
                with torch.no_grad():
                    self.config = AutoConfig.from_pretrained(
                        pretrained_model_path, torchscript=self.torchscript, return_dict=True
                    )

                    self.model = AutoModelForCausalLM.from_pretrained(
                        pretrained_model_name_or_path=pretrained_model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.half,
                        config=self.config,
                    )

        if self.torchscript:
            self.model = torch.jit.script(self.model)

        self.model = self.model.to(device=self.device, dtype=self.dtype)

        self.tokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_path, use_fast=False, torchscript=self.torchscript
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def generate(self, input_str, max_new_tokens) -> Tuple:
        inputs = self.tokenizer(input_str, return_tensors="pt").to(device=self.device)

        input_ids = inputs.input_ids
        seq_len = input_ids.shape[1]

        cache_position = torch.arange(seq_len, dtype=torch.int64, device=self.device)
        generated_ids = torch.zeros(1, seq_len + max_new_tokens * 2, dtype=torch.int, device=self.device)
        generated_ids[:, cache_position] = input_ids.to(self.device).to(torch.int)

        past_key_values = StaticCache(
            self.model.config, 1, seq_len + max_new_tokens * 2 + 1, device=self.device, dtype=torch.float16
        )

        logits = self.model(
            input_ids, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0]
        next_token = torch.argmax(logits[:, [-1]], dim=-1).to(torch.int)
        generated_ids[:, [seq_len]] = next_token

        torch._dynamo.config.capture_scalar_outputs = True

        forward_time_s = []
        with torch.no_grad():
            # Compile the CUDA graph
            if self.backend is None:
                decode_one_tokens_compiled = decode_one_tokens
            else:
                decode_one_tokens_compiled = torch.compile(decode_one_tokens, mode="default", backend=self.backend)

            # Generate tokens one by one
            cache_position = torch.tensor([seq_len + 1], device="cuda")
            for _ in range(1, max_new_tokens):
                nxt = next_token.clone()
                start_time = time.time()
                next_token = decode_one_tokens_compiled(
                    self.model, nxt, None, cache_position, past_key_values
                )
                end_time = time.time()
                generated_ids[:, cache_position] = next_token.int()
                print(f"duration = {end_time - start_time}")
                forward_time_s.append(end_time - start_time)

                cache_position += 1

        return self.tokenizer.decode(generated_ids[0]), forward_time_s



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
                print(tensor_path)
                if is_legacy:
                    ds_legacy = load_legacy_tensor(tensor_path)
                    ds_module = SparsifiedLinear.from_legacy(ds_legacy, 'cpu')
                else:
                    ds_module = torch.load(tensor_path, 'cpu', weights_only=False)
                    """
                    try: 
                    except:
                        print('skipping')
                        continue
                    """
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

    replace_and_save_quantized_layers(model, args.tensors_path, is_legacy)

    torch.save(model, args.torch_pt_path)

    #
    # m = Mode(2)
    #
    # with torch.no_grad():
    #     model = InferenceDemo(args.base_model, None, m, model=model)
    #     text = "Valkyria Chronicles is a video game "  # input()
    #     s = time.time()
    #     generated_text, timings_s = model.generate(text, max_new_tokens=32)
    #     e = time.time()
    #     print(f"{generated_text}")
    #
    #     print(f"Total duration = {e - s}s")
    #
    #     durations = np.array(timings_s[16:])
    #
    #     print(f"Mean duration after caching initial input = {durations.mean()}")
    #     print(f"Median duration after caching initial input = {np.median(durations)}")
    #     print(f"Best duration after caching initial input = {np.min(durations)}")
