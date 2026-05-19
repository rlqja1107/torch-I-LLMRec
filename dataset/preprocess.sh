# Dataset: Grocery, Sport, Phone, Art


data_name=$1 # Art
python dataset/download_and_preprocess.py --data_name $data_name
python dataset/down_image.py --data_name $data_name
python dataset/filter_img_noise.py --data_name $data_name
python dataset/neg_sampling_and_data4sasrec.py --data_name $data_name

