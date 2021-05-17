本文档解释了张炅焱的毕业设计中所写的代码。代码按照顺序撰写，从处理数据到分析数据再到成图。
代码应当在并行运算环境和配置Nvidia显卡的环境运行，并安装对应的CUDA。
按照代码文件执行顺序，给每个文件以相应解释。
1. mergeData.py             按照日粒度处理航空网络数据
2. monthHandle.py           按照月粒度处理航空网络数据
3. Weighted_Months.py       按照月粒度计算航空网络边权值
4. download.py              下载GDELT数据
5. batchHandle.py           整合航空网络和GDELT数据
6. CommunityEvaluation.py   社区划分算法的评估
7. EdMot.py                 采用EdMot法计算社区
8. degreeAndHotness.py      计算航空节点权值
9. IndicatorsForWhole.py    计算航空网络全局指标
10. airports.py             计算高排位机场
11. graph.py                绘制Sigmoid和ReLu图形
12. LinearLearning.py       线性层分析航空网络和GDELT关系
13. RNN.py                  用RNN分析航空网络和GDELT关系

请将命令行路径移入到本文件夹后，使用pip install -r requirements.txt下载所需的依赖包。
然后依次运行代码文件。

请安装CUDA和CuDNN，以支持Pytorch的运行。请在https://pytorch.org/ 选取合适的命令行进行pytoch安装。具体安装流程请见https://zhuanlan.zhihu.com/p/88903659
计算机应当配置N卡，如没有N卡或者难以进行软件配置，请删除pytorch代码中所有to_device语句。

数据说明：
1. GDELT数据，请见http://data.gdeltproject.org/events/index.html，具体数据说明请见https://www.gdeltproject.org/data.html#rawdatafiles，说明了事件网络的多种属性；
2. 航空网络数据，请见https://zenodo.org/record/4737390#.YKHG7agzZPY。数据包含的各种属性请也参见此页面；
3. 疫情数据，请见https://github.com/CSSEGISandData/COVID-19，这是约翰霍普金斯大学所收集的数据，包含全世界各地的确诊量，死亡量和治愈量等。
