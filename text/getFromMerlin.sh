#! /bin/bash

VERBOSE=""
LOGIN="xpeska05"

if [ "$1" == "-v" ]; then
	VERBOSE="-v"
fi

#scp -v xpeska05@merlin.fit.vutbr.cz:~/Projekty/ITY/dokument.pdf .
scp ${VERBOSE} ${LOGIN}@eva.fit.vutbr.cz:~/Bakalarka/projekt.pdf .
# scp ${VERBOSE} ${LOGIN}@eva.fit.vutbr.cz:~/Bakalarka/projekt.out .
# scp ${VERBOSE} ${LOGIN}@eva.fit.vutbr.cz:~/Bakalarka/projekt.ptc .
# scp ${VERBOSE} ${LOGIN}@eva.fit.vutbr.cz:~/Bakalarka/projekt.toc .
