.PHONY: pdf clean figures

pdf: main.pdf

main.pdf: main.tex refs.bib figures/cycle_trace.pdf
	tectonic main.tex

figures/cycle_trace.pdf:
	mkdir -p figures
	uv run python ../../scripts/render_paper_figures.py

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
