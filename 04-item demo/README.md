该目录保存的是使用accelerator进行分布是训练的模板，并且进行了详细注释<br/>
1、包括accelerator的初始化，<br/>
2、使用混合精度<br/>
3、使用梯度累计功能<br/>
4、使用tensorboard实验记录工具<br/>
5、数据集的定义<br/>
6、模型保存<br/>
7、训练状态的保存<br/>
8、断点续训<br/>
9、args的定义<br/>
如果使用diffuser的模型进行训练，那么数据集的定义就是json格式[{"image_file": file_name, "text": prompt}]
