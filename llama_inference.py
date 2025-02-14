import os
import sys
import time

import torch
import torch.nn as nn

from doublesparse import *
from modelutils import *
from datautils import *

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')  # , cache_dir='/scratch/p490-24-t/llamas')
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev, save):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": torch.arange(model.seqlen).to(device=dev)}
    position_ids = torch.arange(model.seqlen).to(device=dev).unsqueeze(0)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    if args.fix_mask:
        masks = {}
        for n, p in model.named_parameters():
            if "layers" in n and "weight" in n and len(p.shape) == 2:
                shape_key = min(p.shape), max(p.shape)
                if shape_key in masks:
                    continue
                dim = shape_key[0]
                nnz = 0.1 if shape_key[0] == shape_key[1] else 0.2
                print(n, p.shape, shape_key, nnz)
                A = torch.eye(dim, device="cuda")
                Arand = torch.rand_like(A)
                Arand += A * 100
                thres = Arand.abs().flatten().sort()[0][int(A.numel() * (1 - nnz))]
                masks[shape_key] = (Arand.abs() > thres)

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                if (
                        not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue

                fixmask = None
                if args.fix_mask:
                    shape_key = min(subset[name].weight.shape), max(subset[name].weight.shape)
                    fixmask = masks[shape_key]
                gpts[name] = DoubleSparse(subset[name], nofinal=args.no_final, fixmask=fixmask)

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()
            del outs

            for name in subset:
                print(i, name)
                print("Pruning ...")
                sparsity = args.sparsity
                sparsified_linear = gpts[name].fasterprune(
                    sparsity,
                )
                if save:
                    folder_path = os.path.join(save, str(i))
                    os.makedirs(folder_path, exist_ok=True)
                    full_path = os.path.join(folder_path, name)
                    torch.save(sparsified_linear, full_path)
                gpts[name].free()

        outs = torch.zeros_like(inps)
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    position_ids = torch.arange(model.seqlen).to(device=dev).unsqueeze(0)
    cache = {"i": 0, "attention_mask": None, "position_ids": position_ids}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * args.sparsity)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    import pdb; pdb.set_trace() # TODO: Remove

    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    seqlen = 4096
    nsamples = 1
    seed = 0
    dataset = 'wikitext2'
    model = torch.load(sys.argv[1])
    model.eval()
    model = model.to(device=DEV)
    model.seqlen = seqlen

    # model_name = '/mnt/6e3c126c-c6bb-43eb-9d82-1e59b2111688/ecrncevi/Llama-2-7b-hf'
    model_name = sys.argv[2]
    dataloader, testloader = get_loaders(
        dataset, nsamples=nsamples, seed=seed, model=model_name, seqlen=seqlen
    )

    llama_eval(model, testloader, DEV, dataset, log_wandb=False)

