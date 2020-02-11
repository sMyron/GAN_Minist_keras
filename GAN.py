from PIL import Image
from keras.models import Sequential
from keras.layers import Dense,Activation,BatchNormalization,Reshape,UpSampling2D,\
    Conv2D,MaxPooling2D,Flatten
import numpy as  np
from keras.optimizers import SGD
from keras.datasets import mnist
import math
# 定义生成器模型
def Generator_model():
    # 下面搭建生成器的架构，首先导入序贯模型
    model = Sequential()
    # 添加一个全连接层，输入为100维向量，输出为1024维
    model.add(Dense(input_dim=100, output_dim=1024))
    # 添加一个激活函数tanh
    model.add(Activation('tanh'))
    # 添加一个全连接层，输出维128×7×7维度
    model.add(Dense(128*7*7))
    # 添加一个批量归一化层，该层在每个batch上将前一层的激活值重新规范化，即使得其输出均值接近于0，标准差接近于1
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    # Reshape层用来将输入shape转换为特定的shape
    model.add(Reshape((7,7,128),input_shape=(128,7,7)))
    # 2维上采样层，即将数据的行和列分别重复2次
    model.add(UpSampling2D(size=(2,2)))
    # 添加一个2维卷积层，卷积核大小维5×5，激活函数为tanh，共64个卷积核，并采用padding使图像尺寸保持不变
    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2,2)))
    # 卷积核设为1即输出图像的维度
    model.add(Conv2D(1,(5,5),padding='same'))
    model.add(Activation('tanh'))
    return model

# 定义判别器模型
def Discriminator_model():
    model=Sequential()
    # 添加二维卷积层，卷积核大小为5×5，激活函数为tanh，
    model.add(
        Conv2D(64,(5,5),
        padding='same',
        input_shape=(28,28,1))
    )
    model.add(Activation('tanh'))
    #添加最大池化层，pool_size取（2，2）使得图片在原来维度上变为一半
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,(5,5)))
    model.add(Activation('tanh'))
    #卷积层过渡到全连接层
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    #一个节点进行二值分类
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

#将生成器和判别器连结成为GAN模型
def GAN_model(g,d):
    model=Sequential()
    #先添加Generator，再令Discriminator不可训练（即固定d）
    model.add(g)
    d.trainable=False
    model.add(d)
    return model

#生成图片进行拼接
def Combine_image(generated_images):
    num = generated_images.shape[0]#图片数目
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]#图片形状取其1-3维，第一维是num
    image = np.zeros((height*shape[0],width*shape[1]),
                     dtype = generated_images.dtype)
    for index,img in enumerate(generated_images):
        i = int(index/width)
        j = index%width
        image[i*shape[0]:(i+1)*shape[0],j*shape[1]:(j+1)*shape[1]]=img[:,:,0]
    return image

#训练GAN
def Train(Batch_size):
    (X_train,y_train),(X_test,y_test)=mnist.load_data()
    #归一化
    X_train = (X_train.astype(np.float32)-127.5)/127.5
    X_train = X_train[:,:,:,None]
    x_test = X_test[:,:,:,None]
    #将定义好的Generator和Discriminator赋值给特定变量
    g = Generator_model()
    d = Discriminator_model()
    d_on_g = GAN_model(g,d)
    #利用SGD优化器
    d_optim = SGD(lr=0.001,momentum=0.9,nesterov=True)
    g_optim = SGD(lr=0.001,momentum=0.9,nesterov=True)
    #对G、D和GAN进行编译
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    #前一个架构再固定D的情况下训练了G，故此处需先设置D为可训练
    d.trainable = True
    d.compile(loss='binary_crossentropy',optimizer=d_optim)

    for epoch in range(30):
        print("Epoch is",epoch)
        #计算一个epoch所需要的迭代次数。
        print("Number of batches is ",int(X_train.shape[0]/Batch_size))

        #一个epoch内进行迭代训练
        for index in range(int(X_train.shape[0]/Batch_size)):
            # 随机生成均匀分布，上下边界为1和-1，输出Batch_size×100个样本
            noise = np.random.uniform(-1,1,size=(Batch_size,100))
            #抽取一个批量的真实照片
            image_batch = X_train[index * Batch_size:(index + 1) * Batch_size]
            #生成的图片使用G对随机噪声进行推断
            generated_images = g.predict(noise, verbose=0)

            #每经过100次迭代输出一张生成的图片
            if index % 100 == 0:
                image = Combine_image(generated_images)
                #逆归一化
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    "./GAN_Picture_train/" + str(epoch) + "_"+ str(index) + ".png"
                )

            #将真实的图片和生成的图片以数组的形式拼接再一起，真实图片在上，生成图片在下
            X = np.concatenate((image_batch,generated_images))
            #生成真假标签，即一个包含两倍批量大小的列表，前一个批量均为1，代表真，后一个批量均为0，代表生成图片
            y = [1] * Batch_size + [0] * Batch_size

            #判别器的损失；在一个batch的数据上进行一次参数更新
            d_loss = d.train_on_batch(X,y)
            print("batch %d d_loss : %f" % (index,d_loss))
            #随机生成均匀分布噪声
            noise = np.random.uniform(-1,1,(Batch_size,100))
            #固定D
            d.trainable = False
            #计算生成器损失；在一个batch的数据上进行一次参数更新
            g_loss = d_on_g.train_on_batch(noise,[1]*Batch_size)
            #令D可训练
            d.trainable = True
            print("batch %d g_loss : %f" %(index,g_loss))
            #每100次迭代保存一次G和D的权重
            if index%100 == 9:
                g.save_weights('generator_weight',True)
                d.save_weights('discriminator_weight',True)

#产生图片
def Generate(Batch_size,nice=False):
    #调用Generator
    g = Generator_model()
    g.compile(loss='binary_crossentropy',optimizer="SGD")
    #加载权重
    g.load_weights('generator_weight')
    if nice:
        d = Discriminator_model()
        d.compile(loss='binary_crossentropy',optimizer="SGD")
        d.load_weights('disriminator_weight')
        noise = np.random.uniform(-1,1,(Batch_size*20,100))
        generated_images = g.predict(noise,verbose=1)
        d_pret = d.predict(generated_images,verbose=1)
        #产生0到Batch_size*20的数，步长为1
        index =np.arange(0,Batch_size*20)
        #转换形状
        index.resize((Batch_size*20,1))
        pre_with_index =list(np.append(d_pret,index,axis=1))#axis=1表示沿着行向量的方向
        pre_with_index.sort(key=lambda x:x[0],reverse=True)#reverse=True表示升序；key=lambda x:x[0]指明所根据的字段进行排序
        nice_images = np.zeros((Batch_size,)+generated_images.shape[1:3],dtype=np.float32)
        nice_images = nice_images[:,:,:,None]
        for i in range(Batch_size):
            idx = int(pre_with_index[i][1])
            nice_images[i,:,:,0] = generated_images[idx,:,:,0]
        image = Combine_image(nice_images)
    else:
        #随机生成正态分布噪声
        noise = np.random.uniform(-1, 1, (Batch_size, 100))
        generated_images = g.predict(noise, verbose=0)
        image = Combine_image(generated_images)
    image = image*127.5+127.5
    #保存图片
    Image.fromarray(image.astype(np.uint8)).save(
        "./GAN_Picture/Generate_image3.png"
    )
# Train(100)
Generate(100)


