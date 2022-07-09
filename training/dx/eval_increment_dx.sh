for input_type in norm unnorm
do
  for n_partials in 1 2 4
  do
    for model_metric in macro_f1 micro_f1 cohen_kappa mcc hat3 hat5 hat8
    do
      echo "Evaluating input_type = ${input_type} / n_partials = ${n_partials} / model_metric = ${model_metric}"
      python eval_increment_dx.py \
        --args_path ./models_increment/encoder-BioLinkBERT__optimizer-AdamW__scheduler-linear__lr-5e-05__n_partials-${n_partials}__input_type-${input_type}__label_type-outicd__scheme-everyk/args.pickle \
        --model_metric ${model_metric}
    done
  done
done