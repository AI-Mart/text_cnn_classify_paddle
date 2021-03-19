# text_cnn_classify_paddle
 基于PaddlePaddletext_cnn文本分类\n
一、更新位置 \n
1 所有文件的路径/home/aistudio/data 改成./data/ \n
2 train.py 64行增加paddle.enable_static()#####add by mart 21-3-19 \n
3 predict.py 8行增加paddle.enable_static()#####add by mart 21-3-19 \n
4 predict.py 35行改成data.append(np.int64(dict_txt[s]))#change by mart 21-3-19 \n

二、 \n
data.py 生产字典构造数据集\n
train.py 训练模型\n
predict.py 基于训练好的模型进行预测\n
data 存放数据集的文件夹\n
work 存放模型的文件夹\n

三、\n
运行predict.py结果示例\n
预测结果标签为：0， 名称为：文化， 概率为：0.906921\n
预测结果标签为：8， 名称为：国际， 概率为：0.488133\n
