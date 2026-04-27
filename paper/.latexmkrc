#!/usr/bin/env perl
$pdf_mode = 5;                # xelatex
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex = 'bibtex %O %B';
$biber = 'biber %O --bblencoding=utf8 -u -U --output_safechars %B';
$makeindex = 'makeindex %O -o %D %S';
$max_repeat = 5;
