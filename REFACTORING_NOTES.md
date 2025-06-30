# 重构说明：将解析器从 PLY 改为 ANTLR 4

## 概述

本次重构的主要目标是将波形表达式解析器从 PLY (Python Lex-Yacc) 改为 ANTLR 4，以提供更好的语法分析能力和错误处理。

## 主要变化

### 1. 新增 ANTLR 4 支持

- 创建了 `waveforms/Waveform.g4` 语法文件，定义了波形表达式的语法规则
- 创建了 `waveforms/antlr_parser.py` 模块，包含基于 ANTLR 4 的解析器实现
- 添加了 `antlr4-python3-runtime` 依赖

### 2. 重构 `wave_eval` 函数 

- 修改了 `waveforms/waveform_parser.py` 中的 `wave_eval` 函数
- 新增了 `use_antlr` 参数，默认为 `True`
- 实现了智能回退机制：如果 ANTLR 解析失败，会自动尝试使用 PLY 解析器

### 3. 保持向后兼容性

- 保留了原有的 PLY 解析器作为回退方案
- 所有现有的 API 保持不变
- 添加了 `ply>=3.11` 作为备选依赖，确保回退功能正常

### 4. 改进的 GitHub Actions

- 更新了 `.github/workflows/workflow.yml`
- 添加了 Java 环境设置和 ANTLR 工具安装
- 优化了多平台构建和发布流程
- 分离了测试和发布阶段，提高了构建效率

## 使用方式

### 默认使用 ANTLR 4 解析器

```python
from waveforms import wave_eval

# 默认使用 ANTLR 4 解析器
waveform = wave_eval("gaussian(10) * cos(2*pi*50)")
```

### 强制使用 PLY 解析器

```python
from waveforms import wave_eval

# 强制使用传统的 PLY 解析器
waveform = wave_eval("gaussian(10) * cos(2*pi*50)", use_antlr=False)
```

## 当前状态

**注意**: ANTLR 4 解析器的完整实现尚未完成。目前系统会自动回退到 PLY 解析器，确保所有功能正常工作。

- ✅ PLY 回退机制正常工作
- ✅ 所有现有测试通过
- ✅ 向后兼容性保持
- 🔄 ANTLR 4 解析器实现进行中

## 测试结果

所有现有测试都通过了，确保重构没有破坏任何现有功能：

```
20 tests passed
```

## 下一步计划

1. 完成 ANTLR 4 解析器的完整实现
2. 添加 ANTLR 特定的测试用例
3. 优化错误处理和错误信息
4. 在确认 ANTLR 解析器稳定后，逐步弃用 PLY 解析器

## 开发者注意事项

- 如果需要修改语法规则，请编辑 `waveforms/Waveform.g4` 文件
- 生成 ANTLR 解析器需要 Java 环境和 ANTLR 4 工具
- CI/CD 管道已经配置了自动生成 ANTLR 文件的功能 