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
通过K线开盘、高点、低点、收盘描述行情怎么波动。
```

成交量：

```text
通过当前K线成交量与上一根K线成交量的变化描述参与度变化。
```

持仓量：

```text
通过当前K线持仓量与上一根K线持仓量的变化描述多空双方态度。
```

时间：

```text
每个周期只使用本周期K线时间描述行情。
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
1. 周线K线图
2. 周线客观描述
3. 周线K线数据
4. 日线K线图
5. 日线客观描述
6. 日线K线数据
7. 小时线K线图
8. 小时线客观描述
9. 小时线K线数据
```
