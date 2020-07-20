all: compile clean #view

compile:
	pdflatex main.tex
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -f *.bbl *.blg *.out *.aux *.log *.fls *.sta *.log *.fdb_latexmk *.toc *.pdf 2>/dev/null

install: 
	sudo apt install aspell texlive
spell:
	sh spellcheck.sh
view:
	okular main.pdf & exit

