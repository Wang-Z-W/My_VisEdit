cd ..
python vead_train.py \
    -dvc "6" \
    -edvc "7" \
    -mn blip2 \
    -dna WaterBird \
    -dfn edit_annotations_truelabel_balanced \
    -spt train \
    -dn 9999999 \
    -bs 4 \
    -eps 80