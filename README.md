# text_cnn_classify_paddle
 基于PaddlePaddletext_cnn文本分类\
一、更新位置 \
1 所有文件的路径/home/aistudio/data 改成./data/ \
2 train.py 64行增加paddle.enable_static()#####add by mart 21-3-19 \
3 predict.py 8行增加paddle.enable_static()#####add by mart 21-3-19 \
4 predict.py 35行改成data.append(np.int64(dict_txt[s]))#change by mart 21-3-19 

二、 \
data.py 生产字典构造数据集\
train.py 训练模型\
predict.py 基于训练好的模型进行预测\
data 存放数据集的文件夹\
work 存放模型的文件夹

三、\
运行predict.py结果示例\
预测结果标签为：0， 名称为：文化， 概率为：0.906921\
预测结果标签为：8， 名称为：国际， 概率为：0.488133
