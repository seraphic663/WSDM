# WSDM 2027 Short LaTeX Baseline

This directory contains a shared ACM `sigconf` wrapper, the current four-page author-visible empirical draft, and a minimal review-body skeleton.

- `template.tex`: shared wrapper. With `\PaperArxivMode` defined it builds an author-visible `sigconf,nonacm` arXiv draft; without that flag it uses the mandatory anonymous WSDM review class. It owns conference metadata, author rendering, CCS behavior, title rendering, Ethical Considerations placement, and bibliography placement.
- `template-body.tex`: minimal body skeleton used when compiling `template.tex` directly.
- `demo.tex`: thin entry file that currently selects arXiv mode and defines `Ziming Zhao`, `seraphic221@outlook.com`, `Renmin University of China`, `Beijing, China`, title, abstract, keywords, body, ethics, and bibliography inputs before loading `template.tex`.
- `demo-body.tex`: current empirical paper body reporting the Stage 0/1 negative-reliability audit and PIVOT gate.
- `demo.bib`: references used by the current paper.
- `demo.pdf`: verified four-page author-visible output of `demo.tex`; pages 1--3 contain the self-contained paper and page 4 contains Ethical Considerations and references.

The mandatory review class is:

```latex
\documentclass[sigconf,anonymous,review]{acmart}
```

Format changes made in `template.tex` automatically propagate to `demo.tex`. Project prose, author identity, and experiment-specific macros belong in the entry/body files, not in the shared template.

For the future WSDM review entry, omit `\PaperArxivMode` and all identity macros; the shared template will then use `\documentclass[sigconf,anonymous,review]{acmart}`.

Compile from this directory with `latexmk` when Perl is available:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error demo.tex
```

The current verified Windows/MiKTeX fallback does not require Perl:

```text
pdflatex -interaction=nonstopmode -halt-on-error demo.tex
bibtex demo
pdflatex -interaction=nonstopmode -halt-on-error demo.tex
pdflatex -interaction=nonstopmode -halt-on-error demo.tex
```

The WSDM limit is four pages for all main content, including figures, tables, and appendices. References and the Ethical Considerations section may continue beyond those four pages. Supplementary material is optional and cannot replace a self-contained paper.
