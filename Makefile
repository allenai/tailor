NLTK_DOWNLOAD_CMD = python -c 'import nltk; [nltk.download(p) for p in ["stopwords"]]'

.PHONY : docs
docs :
	@cd docs && make html && open build/html/index.html

.PHONY : download-extras
download-extras :
	$(NLTK_DOWNLOAD_CMD)