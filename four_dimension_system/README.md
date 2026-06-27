# 螺纹钢三周期客观行情描述系统

## 目标

只描述行情本身。

标的：

```text
螺纹钢 RB0
```

三个周期：

```text
周线
日线
小时线
```

四个维度：

```text
价格
成交量
持仓量
时间
```

## 核心原则

```text
周线、日线、小时线分别独立观察。
三个周期不互相影响。
三个周期不互相作为判断依据。
```

## 描述内容

价格：

```text
通过K线高点、低点、收盘位置描述行情怎么波动。
```

成交量：

```text
通过放量、缩量、量能平稳描述多空双方是否主动参与。
```

持仓量：

```text
通过增仓、减仓、持仓平稳描述多空双方态度和波动持续性。
```

时间：

```text
通过波动扩张、波动收缩、节奏平稳描述行情波动节奏。
```

## 运行

```bash
cd four_dimension_system
python build_site.py
```

打开：

```text
four_dimension_system/web/index.html
```

## 数据字段

```text
datetime
symbol
open
high
low
close
volume
open_interest
```

## 页面输出

```text
1. 三周期 × 四维度行情矩阵
2. 周线客观描述
3. 日线客观描述
4. 小时线客观描述
5. 小时收盘路径图
6. 周线K线数据
7. 日线K线数据
8. 小时线K线数据
```
