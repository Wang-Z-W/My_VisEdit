cd ..
python vead_train.py \
    -dvc "1" \
    -edvc "2" \
    -mn llava \
    -dna WaterBird \
    -dfn edit_annotations_truelabel_balanced \
    -spt train \
    -dn 9999999 \
    -bs 4 \
    -eps 80