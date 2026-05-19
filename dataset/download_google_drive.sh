
data=$1
mkdir -p dataset/Amazon_18
if [ $data = "Art" ]; then
    gdown "https://drive.google.com/uc?id=1_sraTCmJRNZ9TgkrVOqur45S-yPMGxMA"
    mv Art.zip dataset/Amazon_18
    cd dataset/Amazon_18
    unzip Art.zip
    rm -rf Art.zip
elif [ $data = "Sport" ]; then 
    gdown "https://drive.google.com/uc?id=1oIgUzma1ZB9S2aCHADFB8lY1jZ5VEHwV"
    mv Sport.zip dataset/Amazon_18
    cd dataset/Amazon_18
    unzip Sport.zip
    rm -rf Sport.zip
elif [ $data = "Phone" ]; then 
    gdown "https://drive.google.com/uc?id=14MGnbTDnDcXgUJ3YzdqEtbJYOUyFWxCQ"
    mv Phone.zip dataset/Amazon_18
    cd dataset/Amazon_18
    unzip Phone.zip
    rm -rf Phone.zip
elif [ $data = "Grocery" ]; then 
    gdown "https://drive.google.com/uc?id=1Iy6vp7LpKgVJ6raYrZBpqZANi9zaF9u0"
    mv Grocery.zip dataset/Amazon_18
    cd dataset/Amazon_18
    unzip Grocery.zip
    rm -rf Grocery.zip
fi
