#! /bin/bash

FILES="proj3.tex Makefile"
LOGIN="xpeska05"
VERBOSE=""

zip -r bakalarka.zip *
FILES="bakalarka.zip"

if [ "$1" == "-v" ]; then
	VERBOSE="-v"
fi

echo "Zkousim odeslat soubory pres server merlin."
scp ${VERBOSE} ${FILES} ${LOGIN}@eva.fit.vutbr.cz:~/Bakalarka/

if [ $? == 1 ]; then
	echo "Zkousim odeslat soubory pres server eva."
	scp ${VERBOSE} ${FILES} ${LOGIN}@eva.fit.vutbr.cz:~/Bakalarka/
else
	echo "Odeslani uspesne dokonceno."
	exit
fi

if [ $? -eq 1 ]; then
	echo "Soubor se nepodarilo odeslat."
	exit
fi
