for LEARNING_RATE in 2e-5
    do
        for image_feature in resnet50
            do
                for triple_number in 5
                    do
                        CUDA_VISIBLE_DEVICES=1 python model/main_bart.py \
                        --BATCH_SIZE 16 \
                        --EPOCHS 30 \
                        --LEARNING_RATE ${LEARNING_RATE} \
                        --RANDOM_SEEDS 42 \
                        --image_feature ${image_feature} \
                        --dataset twitter2015 \
                        --triple_number ${triple_number}
                    done
            done
    done