import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import cv2
import models
import numpy as np
import argparse
import random
from sklearn import svm
#-------------------------------------------------------
# lr = 0.0001
# epochs = 20
# batch_size = 2
# dropout_rate = 0 #0为不开启丢弃学习
# lr_reduce = False
# l2_rate = 0 #0为不开启l2正则化权重衰减
#-------------------------------------------------------
# 添加高斯噪声
def add_noise(img):
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

# 数据集生成
def dataGenerate(batch_size,data_enhancement):
    # 是否数据增强
    if(data_enhancement):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range = 20,            #旋转
            width_shift_range = 3,          #横向平移
            height_shift_range = 5,         #纵向平移
            zca_whitening = True,           #ZCA化
            preprocessing_function=add_noise,#噪声添加
            rescale=1./255,                 
            validation_split=0.2)
    else:
         datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2)

    train_generator = datagen.flow_from_directory(
            # 种类文件夹的路径
            'characterData',
            # 目标图片大小
            target_size=(20,20),
            # 目标颜色模式
            color_mode="grayscale",
            # 种类名字
            classes=None,
            # 种类模式：分类
            class_mode='categorical',
            # batch_size
            batch_size=batch_size,
            # shuffle
            shuffle=True,
            # seed
            seed=None,
            # # 变换后的保存路径
            save_to_dir=None,
            # 保存的前缀
            save_prefix="c",
            # 保存的格式
            save_format="png",
            # 验证分离的设置
            subset="training"
    )

    val_generator = datagen.flow_from_directory(
            # 种类文件夹的路径
            'characterData',
            # 目标图片大小
            target_size=(20,20),
            # 目标颜色模式
            color_mode="grayscale",
            # 种类名字
            classes=None,
            # 种类模式：分类
            class_mode='categorical',
            # batch_size
            batch_size=batch_size,
            # shuffle
            shuffle=True,
            # seed
            seed=None,
            # # 变换后的保存路径
            save_to_dir=None,
            # 保存的前缀
            save_prefix="c",
            # 保存的格式
            save_format="png",
            # 验证分离的设置
            subset="validation"
    )
    return train_generator, val_generator

    

def write_weight(label_index):
    if(not os.path.exists('weights')):
        os.mkdir('weights')

    with open('weights/classes.csv', 'w', encoding='utf_8_sig',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['classes'])
        for c in label_index:
            writer.writerow([c])
# 深度学习模型
def train_DLmodel(model_name, train_generator,val_generator,lr,epochs,dropout_rate,lr_reduce,l2_rate):
    if(model_name=='CNN'):
        model = models.CNN(l2_rate, dropout_rate)
    elif(model_name == 'AlexNet'):
        model = models.AlexNet()
    elif(model_name == 'ResNet'):
        #网络参数设置
        model = models.ResNet([2, 2, 2], 31)
        model.build(input_shape=(None, 20, 20, 1))
    print(model.summary())
    # 编译模型
    model.compile(optimizer=tf.optimizers.Adam(lr),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    #回调函数

    cp = keras.callbacks.ModelCheckpoint('weights/best',
                                                save_best_only=False,
                                                save_weights_only=True)

    lrr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',  # 学习率衰减回调函数
                                                factor=0.1, 
                                                patience=5, 
                                                verbose=1, 
                                                min_delta=0.0001)

    callbacks = [cp]
    if lr_reduce :
        callbacks.append(lrr)

    # 拟合数据
    history = model.fit(
                train_generator,
                steps_per_epoch=None,
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=None,
                callbacks=callbacks)
    return history, model


# 机器学习模型
def train_MLmodel(model_name, train_generator,val_generator):
    if(model_name == 'MLP'):
        model =MLPClassifier(hidden_layer_sizes=(50,),  alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-4, random_state=1, learning_rate='invscaling',
                    learning_rate_init=.0001, power_t=0.001)
    elif(model_name == 'SVM'):
        model =svm.SVC(gamma=0.001, C=100., probability=True)

    X_train,y_train = models.getImage(train_generator, train_generator.labels)
    X_val, y_val = models.getImage(val_generator, val_generator.labels)
    history = model.fit(X_train, y_train)
    print("Training set score: %f" % model.score(X_train, y_train))
    print("Validation set score: %f" % model.score(X_val, y_val))
    return history, model

def plot_figure(history):
    plt.plot(history.epoch,history.history['loss'], label='train_loss')
    plt.plot(history.epoch,history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig('weights/loss.jpg')
    plt.show()
    plt
    plt.plot(history.epoch,history.history['accuracy'], label='train_accuracy')
    plt.plot(history.epoch,history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.savefig('weights/accuracy.jpg')
    plt.show()


def write_csv(model_type, model, label_index):
    # 写入CSV
    file = 'results.csv'
    results = []

    with open(file, 'w', encoding='utf_8_sig',newline='') as csv_file:
        print('\n--- printing results to file {}'.format(file))
        writer = csv.writer(csv_file)
        writer.writerow(['picture', 'label'])
        predict_path = r'predict'
        picture_list = [f for f in os.listdir(predict_path) if not f.startswith('.')]
        picture_list.sort(key=lambda x: int(x[:-4]))   
        for picture in picture_list:
            img = cv2.imread('predict/' + picture, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (20, 20))
            img_arr = img / 255.0
            if(model_type == 'DL'):
                img_arr = img_arr.reshape((1, 20, 20, 1))
                y = model.predict(img_arr, batch_size=1)
            else:
                img_arr = img_arr.reshape((1, 400))
                y = model.predict(img_arr)
            res = label_index[y.argmax()]
            results.append(res)
            writer.writerow([picture, res])
    # 计算准确率
    ground_truth_csv = r'ground_truth.csv'
    ground_truth_dic = {}
    with open(ground_truth_csv, 'r',encoding='utf-8') as f:
        print('\n--- reading ground truths from file {}'.format(ground_truth_csv))
        reader = csv.reader(f)
        right = 0

        next(reader)

        i = 0
        for line in reader:
            if(results[i] == line[1]):
                right += 1
            i += 1
        
        print('predict accuracy: ', right / len(picture_list))


if __name__ == "__main__":
    # 模型名
    MLmodel_name = ['MLP', 'SVM']
    DLmodel_name = ['CNN', 'AlexNet', 'ResNet']

    parser = argparse.ArgumentParser(description='Province pics classification')
    parser.add_argument('--multimodel', default=0, type=int)
    parser.add_argument('--model_type', default='DL', type=str)
    parser.add_argument('--model_name', default='CNN', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--lr_reduce', default=True, type=bool)
    parser.add_argument('--l2_rate', default=0, type=int)
    parser.add_argument('--data_enhancement', default=True, type=bool)
    args = parser.parse_args()


    train_generator, val_generator = dataGenerate(args.batch_size, args.data_enhancement)
    label_index = list(train_generator.class_indices)
    write_weight(label_index)
    # 多模型训练
    if(args.multimodel==1): 
        # # 训练机器学习模型
        # for mlmodel in MLmodel_name:
        #     train_MLmodel(mlmodel)
        
        # 训练深度学习模型
        for dlmodel in DLmodel_name:
            print(dlmodel)
            history, model = train_DLmodel(dlmodel, train_generator=train_generator,val_generator=val_generator,lr=args.lr,epochs=args.epochs,dropout_rate=args.dropout_rate,lr_reduce=args.lr_reduce,l2_rate=args.l2_rate)
            plot_figure(history)
            write_csv(model, label_index)
    # 单模型训练
    elif(args.model_type == 'DL'):
        print(args.model_name)
        history, model = train_DLmodel(model_name=args.model_name, train_generator=train_generator,val_generator=val_generator,lr=args.lr,epochs=args.epochs,dropout_rate=args.dropout_rate,lr_reduce=args.lr_reduce,l2_rate=args.l2_rate)
        plot_figure(history)
        write_csv(args.model_type,model, label_index)
    elif(args.model_type == 'ML'):
        history, model = train_MLmodel(model_name=args.model_name, train_generator=train_generator,val_generator=val_generator)
        write_csv(args.model_type, model, label_index)
    
