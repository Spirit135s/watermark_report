# 运行参数

## 1. `watermark.py`

```bash
python course_project/simple_watermark_demo.py \
  --input examples/pic/ori_img.jpeg \
  --output-dir course_project/output \
  --watermark SignalSystem \
  --d1 40 \
  --d2 4
```

| 参数 | 种类 | 功能介绍 |
| --- | --- | --- |
| `--input` | `Path` | 基础图像地址 |
| `--output-dir` | `Path` | 输出地址（输出文件为`watermarked.png` 和 `result.txt`） |
| `--watermark` | `str` | 待嵌入水印信息 |
| `--d1` | `float` | 见说明 |
| `--d2` | `float` | Secondary embedding strength for the second singular value. Set to `0` to disable the second channel. |

输出：
- `watermarked.png`
- `result.txt`

## 2. `robustness_test.py`

```bash
python course_project/robustness_test.py \
  --input  \
  --output-dir  \
  --watermark  \
  --d1 40 \
  --d2 4 \
  --seed 2026
```

| 参数 | 种类功能介绍 |                                                              |
| --- | --- | --- |
| `--input` | `Path` | 基础图像地址 |
| `--output-dir` | `Path` | 输出地址（ `watermarked.png`，`robustness_results.csv`和`attacked_images/`） |
| `--watermark` | `str` | 待嵌入水印信息 |
| `--d1` | `float` | 同 `watermark.py`. |
| `--d2` | `float` | 同 `watermark.py`. |
| `--seed` | `int` | 随机生成种子 |

输出：
- `watermarked.png`
- `robustness_results.csv`
- `attacked_images/`
