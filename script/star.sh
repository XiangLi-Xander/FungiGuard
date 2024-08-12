#!/bin/bash
#
#pipeline
#Author:Li Xiang(1365697070@qq.com)


# 查找当前目录下所有以 "Classifier.py" 结尾的 Python 脚本并执行
for script in ./*Classifier.py; do
    if [[ -f "$script" ]]; then
        echo "Running $script"
        python "$script"
    else
        echo "No Classifier.py scripts found."
    fi
done
