pip install -r requirements.txt
for i in {0..4}
do
  for temp in 0 0.5 1
  do
    for subset in "humaneval" "mbpp"
    do
      echo "Running inference with seed $i"
      # Set the seed for reproducibility
      PYTHONPATH=. python3 eval_ts/inference_multiple.py --trace False --seed $i --subset $subset --temp $temp
    done
  done
done


