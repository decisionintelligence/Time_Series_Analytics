# 使用说明

## 数据准备
时序预测示例代码使用ETTm2数据集，请在此下载（[link](https://drive.google.com/file/d/1v5az7yXB5J4se5UHrmXzedSCrMlDWAtH/view?usp=sharing)），并解压放置于 `chap3_forecasting/data/`目录。

## 模型说明
### 时序预测
`chap3_forecasting/models/`: 该目录下存放的是模型文件，可以使用该目录下的CNN、Linear、RNN、TCN、Transformer等模型进行时序预测。

## 环境准备
```bash
pip install -r requirements.txt
```
## 示例代码使用

```python
python chap3_forecasting/models/Linear.py
```
