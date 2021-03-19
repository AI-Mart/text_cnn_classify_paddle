import os
from multiprocessing import cpu_count
import numpy as np
import shutil
import paddle
import paddle.fluid as fluid

paddle.enable_static()#####add by mart 21-3-19
# 用训练好的模型进行预测并输出预测结果
# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

save_path = './work/infer_model/'

# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)


# 获取数据
def get_data(sentence):
    # 读取数据字典
    with open('./data/dict_txt.txt', 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    dict_txt = dict(dict_txt)
    # 把字符串数据转换成列表数据
    keys = dict_txt.keys()
    data = []
    for s in sentence:
        # 判断是否存在未知字符
        if not s in keys:
            s = '<unk>'
        # data.append(int(dict_txt[s]))
        data.append(np.int64(dict_txt[s]))#change by mart 21-3-19
    return data


data = []
# 获取图片数据
data1 = get_data('在获得诺贝尔文学奖7年之后，莫言15日晚间在山西汾阳贾家庄如是说')
data2 = get_data('综合“今日美国”、《世界日报》等当地媒体报道，芝加哥河滨警察局表示，')
data.append(data1)
data.append(data2)

# 获取每句话的单词数量
base_shape = [[len(c) for c in data]]

# 生成预测数据
tensor_words = fluid.create_lod_tensor(data, base_shape, place)

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: tensor_words},
                 fetch_list=target_var)

# 分类名称
names = [ '文化', '娱乐', '体育', '财经','房产', '汽车', '教育', '科技', '国际', '证券']

# 获取结果概率最大的label
for i in range(len(data)):
    lab = np.argsort(result)[0][i][-1]
    print('预测结果标签为：%d， 名称为：%s， 概率为：%f' % (lab, names[lab], result[0][i][lab]))