import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵图形
    :param cm: 混淆矩阵
    :param classes: 标签名称列表
    :param normalize: 是否对混淆矩阵进行归一化
    :param title: 图表标题
    :param cmap: 颜色映射
    :return:
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



# 定义标签
labels = ["婚恋交友", "假冒身份" ,"钓鱼网站", "冒充公检法" , "平台诈骗","招聘兼职" ,"杀猪盘" ,"博彩赌博" ,"信贷理财" ,"刷单诈骗" ]

confusion_matrix = \
[[450  , 2  , 2  , 0  , 1  , 0  , 1  , 0 ,  0 ,  2],
 [  0  ,22  , 0  , 0  , 0  , 0  , 0  , 0 ,  1 ,  1],
 [ 21  , 4  ,15  , 0  , 0  , 0  , 1  , 7 ,  2 ,  6],
 [  5  , 0  , 1  ,18  , 1  , 0  , 1  , 0 ,  0 ,  1],
 [ 27  , 3  , 5  , 0  ,11  , 0  , 1  , 2 ,  0 ,  6],
 [  1  , 0  , 0  , 0  , 0  ,15  , 0  , 0 ,  3 ,  1],
 [  0  , 0  , 1  , 0  , 1  , 0  ,22  , 0 ,  1 ,  1],
 [  2  , 0  , 3  , 0  , 0  , 0  , 3  ,44 ,  0 ,  2],
 [ 27  , 2  , 2  , 2  , 3  , 0  , 0  , 1 ,382 , 10],
 [ 18  ,12  ,31  , 4  , 2  , 0  , 7  , 4 ,  6 ,337]]

plt.rcParams["font.sans-serif"]=["DengXian"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题  

confusion_matrix = np.array(confusion_matrix)
# 绘制混淆矩阵
plot_confusion_matrix(confusion_matrix, classes=labels, normalize=True, title='Confusion matrix')
plt.savefig("/A04/IllegalWebsiteClassifier/fig/confusion_matrix.jpg")
