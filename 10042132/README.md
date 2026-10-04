# MBA-RL 论文修订版 10042132

本目录是原论文的独立副本。编译入口为 `main.tex`，所需章节、图片、参考文献和本地样式文件均已保留。

## 文件说明

- `main.tex`：修订后的英文入口文件，摘要与正文统计口径同步。
- `sections/02_problem_formulation.tex` 至 `05_conclusion.tex`：本次修订的章节。
- `sections/01_introduction.tex`、`06_appendix.tex`：按原文件保留，未修改。
- `论文中文译稿.md`：对应当前英文稿的完整中文译文，包括图表、公式、算法、附录和参考文献。
- `修改说明与检查结果.md`：修改范围、关键口径、检查结果和编译限制。
- `图片文字修改清单.md`：逐图说明需要手动修改的图内文字、符号和布局；图片未修改。
- `修订依据与核验记录.md`：关键指标和旧图表口径的本地证据。

原有修订着色机制保留：`\rev` 使用蓝色，`\newrev` 使用红色。颜色不影响正文内容。

## 编译

在安装了 LaTeX 的环境中，从本目录依次运行：

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

本次会话未提供内置 LaTeX 打开/编译工具，也未在 PATH 及已检查位置找到本地 TeX 编译器。本稿已通过源码静态检查，但尚未实际编译；最终页数、浮动体位置、行溢出和版面需要在编译后确认。旧 README 的“已验证 9 页”不适用于本修订版。

中文 Markdown 的图片使用本目录下的相对路径，请与 `figures` 文件夹一起保留。两张矢量 PDF 图以链接方式提供，公式需要支持 LaTeX 数学的 Markdown 阅读器。
