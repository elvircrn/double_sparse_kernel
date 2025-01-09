for sp in 0.5 0.75; do
   for fm in ""; do
      for final in "--no-final"; do
        echo llama2-7-$sp$fm$final;
        python llama.py meta-llama/Llama-2-7b-hf wikitext2 --sparsity $sp $fm $final --save experiment$sp
      done
    done
done
