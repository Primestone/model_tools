## Stacking
stacking过程如下：

![image](https://img-blog.csdn.net/20170915114447314?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd3N0Y2pm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

model1: 将数分层5折，4份做训练，剩余1份做交叉验证， 训练5次，这样会得到一个完整train的new feature, 预测的test的5个结果取均值，得到test的new feature,这样就得到model1的meta feature1。同理，model2按照该过程得到meta feature2, n个模型可得到n个新的特征，作为下一层模型的输入，经训练后得到预测结果。