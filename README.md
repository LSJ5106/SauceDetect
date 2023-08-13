# ***方便面酱料包漏液检测***

## Update
2023.7(v4.0)
* 结合U型结构网络进行精准的分割，可以直接分割出漏液情况。
* 设计UELIE图像增强方法，可以针对弱光、过曝两种现象分别进行前处理。
* 设计MFFM多尺度特征融合方法，更好地融合深层、浅层的特征信息。
* 数据集扩充至606张。
* 最好的效果FPS可达40（4060笔记本端）

2023.4 （v3.0）
* 提高ROI连接处精准性，数据集数量翻10倍！
* 提高ROI连接处判断AP至99%，提高模型鲁棒性。
* FPS可达13（每分钟支持检测780包）。
* 普通摄像机也可以达到较高的检测精度。

2023.3 （v2.0）
* 结合公开数据集，降低误检率

2023.3 （v1.0）
* 数据集为网络开源的少量酱料包
* 仅支持定位到ROI连接处，AP约60%
* 误检率高


## How to Use
For v4.0
* 训练方法
  * python [train_UELIE_MulFuse_SPSM.py](train_UELIE_MulFuse_SPSM.py)
* 推理方法
  * python [get_miou(UELIE+MFFM+SSAM).py](get_miou%28UELIE%2BMFFM%2BSSAM%29.py)
  * python [predict(UELIE+MFFM+SSAM).py](predict%28UELIE%2BMFFM%2BSSAM%29.py)
  * 权重下载：链接：https://pan.baidu.com/s/1HxKhm0EWulqUiCli6tI5hw?pwd=li1n 



For v3.0
* 在网络良好的场所，用迅雷（推荐）下载压缩包
* 输入密码解压（需要输入2次）
* 运行main.exe
* 点击设置，配置流媒体地址
* 返回主页面，点击流媒体摄像头，开始检测
* Redis数据库传输：list[connection_full_count，connection_leak_count]
* 点击停止检测，并关闭cmd


## Note
v4.0开源