for sp in 0.8; do
   for fm in ""; do
      for final in ""; do
        echo llama2-7-$sp$fm$final;
        python llama.py meta-llama/Llama-2-7b-hf wikitext2 --sparsity $sp $fm $final --save experiment$sp
      done
    done
done
