#! /bin/bash

LOGIN="xpeska05"
VERBOSE=""

if [[ -f "bakalarka.zip" ]]; then
    rm bakalarka.zip
fi

make clean
zip -r bakalarka.zip *
ZIP="bakalarka.zip"

if [ "$1" == "-v" ]; then
	VERBOSE="-v"
fi

echo "Zkousim odeslat soubory pres server merlin."
scp ${VERBOSE} ${ZIP} ${LOGIN}@merlin.fit.vutbr.cz:~/Bakalarka/

if [ $? == 1 ]; then
	echo "Zkousim odeslat soubory pres server eva."
	scp ${VERBOSE} ${ZIP} ${LOGIN}@eva.fit.vutbr.cz:~/Bakalarka/
else
	echo "Odeslani uspesne dokonceno."
	exit
fi

if [ $? -eq 1 ]; then
	echo "Soubor se nepodarilo odeslat."
	exit
fi
