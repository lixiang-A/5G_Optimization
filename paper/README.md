# LaTeX 写作说明

这套 LaTeX 初稿用于在 VS Code 中继续写论文正文。当前目录包含：

- `main.tex`：主论文文件
- `references.bib`：当前已整理的参考文献

## 1. 能不能在 VS Code 里写

可以，而且很适合后续长文维护。

推荐组合：

- VS Code 插件：`LaTeX Workshop`
- TeX 发行版：
  - macOS：`MacTeX`
  - Linux：`TeX Live`
  - Windows：`MiKTeX` 或 `TeX Live`

## 2. 当前环境状态

我在当前工作区里检查过，系统里还没有：

- `xelatex`
- `latexmk`

所以当前仓库已经具备可写作的 `.tex` 文件，但**本机要先装 TeX 发行版，之后才能实际编译**。

## 3. 推荐编译方式

这份稿子使用中文 `ctexart`，推荐用 `xelatex`。

如果装好了 TeX 环境，常见编译流程是：

```bash
cd /Users/liz/Desktop/5G_Optimization/paper
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

如果安装了 `latexmk`，更省心：

```bash
cd /Users/liz/Desktop/5G_Optimization/paper
latexmk -xelatex main.tex
```

## 4. VS Code 中的最小操作

1. 安装 `LaTeX Workshop`
2. 打开 `paper/main.tex`
3. 安装好 TeX 发行版后，按插件默认的 `Build LaTeX project`
4. 如果第一次参考文献没出来，再执行一次编译

## 5. 现在这份稿子写到了什么程度

`main.tex` 当前已经不是空骨架，而是包含了：

- 摘要
- 引言
- 相关工作
- 系统模型与问题表述
- 求解框架
- 实验设置与结果分析
- 讨论
- 结论

其中最需要后续继续补强的部分是：

- 文献综述的最终压缩与措辞精修
- 参数灵敏度分析结果表格
- 图表插入与编号
- 作者信息、单位、基金、图题表题格式

## 6. 与仓库其他文档的关系

建议后续写作时这样配合使用：

- `docs/PAPER_DRAFT.md`：中文思路稿与叙事底稿
- `docs/建模与结果初稿.md`：建模与结果证据底稿
- `docs/参数标定与数据来源说明.md`：参数身份与数据来源措辞依据
- `docs/参数灵敏度分析操作指南.md`：学长跑实验的直接操作说明
- `paper/main.tex`：正式论文正文维护文件

## 7. 一个最实用的建议

后续你们不要在 `.tex` 里边写边想。最好保持这个节奏：

- 先在 `docs/` 里把一段内容想清楚
- 再同步进 `paper/main.tex`

这样写作会稳很多，也不容易把“想法”和“正式表述”混在一起。
