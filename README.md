# ***方便面酱料包漏液检测***

## Update
2023.7(v4.0)
* 结合U型结构网络进行精准的分割，可以直接分割出漏液情况，mIou可达80%以上。
* 设计UELIE图像增强方法，可以针对弱光、过曝两种现象分别进行前处理。
* 设计MFFM多尺度特征融合方法，更好地融合深层、浅层的特征信息。
* 数据集扩充至606张。
* 最好的效果FPS可达40（4060笔记本端）
* Combined with the U-shaped network for accurate division, the leak situation can be directly segmented, and mIoU can reach more than 80%.
* Design the UELIE image enhancement method, which can be processed separately for the two phenomena of weak light and overtime.
* Design MFFM multi-scale feature fusion methods to better integrate deep and shallow feature information.
* Data sets expand to 606.
* The best effect FPS can reach 40 (4060 notebook)

  
2023.4 （v3.0）
* 提高ROI连接处精准性，数据集数量翻10倍！
* 提高ROI连接处判断AP至99%，提高模型鲁棒性。
* FPS可达13（CPU每分钟支持检测780包）。
* 普通摄像机也可以达到较高的检测精度。

2023.3 （v2.0）
* 结合公开数据集，降低误检率

2023.3 （v1.0）
* 数据集为网络开源的少量酱料包
* 仅支持定位到ROI连接处，AP约60%
* 误检率高


## How to Use
For v4.0
* 训练方法（train method）
  * python [train_UELIE_MulFuse_SPSM.py](train_UELIE_MulFuse_SPSM.py)
* 推理方法（inference methods）
  * python [get_miou(UELIE+MFFM+SSAM).py](get_miou%28UELIE%2BMFFM%2BSSAM%29.py)
  * python [predict(UELIE+MFFM+SSAM).py](predict%28UELIE%2BMFFM%2BSSAM%29.py)
  * 权重下载（download weights）：链接：https://pan.baidu.com/s/1HxKhm0EWulqUiCli6tI5hw?pwd=li1n 


For v3.1
* 工业级料包连接处漏液检测，可部署边缘设备。联系邮箱510698367@qq.com
* 运行main.exe
* 点击主程序界面【设置】按钮，输入 Redis 地址、端口、RTSP 流媒体地址后保存。
*  点击主程序界面【连接数据库】按钮，待右下角显示连接成功后，点击【摄像头检
测】按钮，开始检测。数据库信息为 list 数组，第 0 项为料包数量，第 2 项为异常漏液数量。
* 如果检测到漏液，会自动在 result_img 文件夹下生成结果图，图片命名格式为：ssim_connection（例如 result_img\0.30000000_connection.png）。
* 注意：前五张连接处不做检测，用作 SSIM 自适应调节。 
* 可根据现场情况设置检测灵敏度。具体于设置界面滑动 SSIM 灵敏度条即可，默认为 0.45。 
* 使用完成后，可以点击【停止检测】按钮，或直接点击控制框（黑白框）的 X 退出即可。 
* 参数介绍：
  * '--data', type=str, default='newdata/sauce.data'：训练/推理配置文件；
  * '--weights', type=str, default='weights/sauce-500-epoch-0.999927ap-model.pth'：已经训练好的权重路径
  * '--video', type=str, default='datasets/xxx.mp4'：需要推理的视频路径。
  * newdata/train和val：数据集


For v3.0
* 运行main.exe
* 点击设置，配置流媒体地址
* 返回主页面，点击流媒体摄像头，开始检测
* Redis数据库传输：list[connection_full_count，connection_leak_count]
* 点击停止检测，并关闭cmd


## Note
* v4.0 代码、权重、生成图、复现方法开源
* 因数据集未拿到品牌方授权，因此数据集暂不开源，联系作者：510698367@qq.com
* v4.0 code, weights, generating images, and reproduction methods are open source.
* Because the datasets is not authorized by the brand, the datasets is not open for the time being, contact the author: 510698367@qq.com
  
## 注意
* v3 联系510698367@qq.com
* v3常见问题汇总：
  * Q: 点击摄像头后，卡顿一会儿提示“出错！错误发生在视频读取”？
    * A: 该问题的原因是流媒体传输存在问题，请检查推流端/接收端是否均启
动了 ffmpeg 服务。
  * Q: 数据库收不到数据？
    * A: 请检查控制框（黑白框）是否打印“[数字, 数字, 数字]”信息。如果
未打印，请点击主界面的连接数据库按钮。 
  * Q: 点击 setup.exe，显示“系统无法运行该程序” ?
    * A: 该问题的原因可能是 VC 库不存在或者版本过低。请下载 VC 库 2015 版
本及以上。 
  * Q: 调用摄像头，控制框提示“视频路径存在错误” ?
    * A: 该问题的原因是程序检测不到摄像头，请检查电脑摄像头是否存在，是
否能正常使用。
  * Q: 点击离线视频后，视频播放 4 秒左右卡住?
    * A: 该问题属于正常现象，该按钮为开发者测试按钮，目的是调试视频是否
正常，用户无需点击该按钮。出现该问题后直接点击控制框（黑白框）的 X 退出即
可。 
  * Q: 点击连接数据库按钮后，提示“由于目标计算机积极拒绝，无法连接” ?
    * A: 该问题是因为 Redis 服务未安装或未启动。请确保该电脑已经安装步骤
一的 msi 文件。如果仍然出现此问题，请打开系统服务 – Redis - 启动该服务。





