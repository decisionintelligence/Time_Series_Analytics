# 使用说明

## 数据准备
时序预测示例代码使用ETTm2数据集，请在此下载（[link](https://drive.google.com/file/d/1v5az7yXB5J4se5UHrmXzedSCrMlDWAtH/view?usp=sharing)），并解压放置于 `forcasting/data/`目录。

## 模型说明
### 时序预测
`chap3_forcasting/models`: 该目录是存放模型的目录，该目录下可以使用Linear、CNN、RNN、TCN、Transformer等模型进行时间序列预测。

## 环境准备
```bash
pip install -r requirements.txt
```
## 代码使用

### 示例：使用Linear模型预测
```python
python chap3_forcasting/models/Linear.py
```

