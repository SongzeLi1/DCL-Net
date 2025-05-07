if [ "$#" -ne 1 ]; then
    echo "Usage: $0 tianchi_test"
    exit
fi

tianchi_train=$1
test_file=./data/tianchi_test.txt
model_path=./ckpt/mvssnet_tianchi.pt
save_dir=./save_out/
threshold=0.5

python inference.py --model_path $model_path --test_file $test_file --save_dir $save_dir

python evaluate.py --pred_dir $save_dir --model_name $model_path --gt_file $test_file --th $threshold
