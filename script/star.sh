#!/bin/bash
#
#pipeline
#Author:Li Xiang(1365697070@qq.com)


for script in ./*Classifier.py; do
    if [[ -f "$script" ]]; then
        echo "Running $script"
        python "$script"
    else
        echo "No Classifier.py scripts found."
    fi
done
