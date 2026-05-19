export CUDA_VISIBLE_DEVICES="3"
data_name=$1
python feature_preprocess/preprocess_visual_feature.py --data_name $data_name
python feature_preprocess/preprocess_textual_feature.py --data_name $data_name

# Training the SASRec to extract the ID featuers
neg_path="./dataset/Amazon_18/${data_name}/neg_item_set.txt"
python ILLMRec/model/SASRec/main.py --inference_only false --dataset $data_name --neg_item_file_path $neg_path
