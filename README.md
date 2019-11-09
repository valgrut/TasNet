# TasNet
Bakalarska prace (Bachelor thesis) - Speaker separation in time domain.

## Resources k praci
- [TasNet - v1](https://arxiv.org/abs/1809.07454v1) (Puvodni)
- [TasNet - v3](https://arxiv.org/abs/1809.07454) (Nova prace TasNet - 2019 (v3) - zlepseny popis modelu atd.)
- www.deeplearningbook.org
- http://www.jzus.zju.edu.cn/oldversion/opentxt.php?doi=10.1631/FITEE.1700814
- NN kurz https://www.coursera.org/learn/neural-networks-deep-learning

## NN
- [Private - merlin CNN seminar](https://merlin.fit.vutbr.cz/wiki/index.php/CNN_semin%C3%A1%C5%99)

## Tools
- [pytorch](https://pytorch.org/)
- [NumFocus](https://numfocus.org/sponsored-projects)
    - [scipy](https://www.scipy.org/)
    - [NumPy](https://numpy.org/devdocs/user/quickstart.html)
    - [jupyter](https://jupyter.org)
- [tensor-flow](https://www.tensorflow.org/)

## Konzultace
### ut 8. 10. 2019 - Konzultace
- prvni schuze po lete
- mala zmena zadani - vliv hyperparametru na vysledky modelu a jak je zmensit tak, aby to co nejmene ovlivnilo vysledne fungovani site
- **Google colab** VS **skolni cluster**
  - vyhody a nevyhody
    - klikani, omezena doba trenovani, snazsi pristup, lze rozdelit zdrojak a data na git a drive, etc.
    - ssh, automatizace, nutne opravneni k pristupu, pracuje nezavisle na pc, etc.
- domluva na pravidelnych schuzkach - **Utery - cca 18 hod**
- Dalsi kroky:
  - vyzkouset google colab
  - upravit zadani
  - udelat tento zaznam

### ut 15. 10. 2019 - Konzultace
- pozvanka na git
- debata o nove verzi prace
- rady ohledne psani textu (citace, bib)
- dalsi konzultace ve **Stredu 23. 10. v 17 hodin**

### st 23. 10. 2019 - Konzultace
- skipped

### ut 28. 10. 2019 - Konzultace
- rozjel jsem Google colab + Google Drive a GPU notebook
- Error Index out of bound: nalezena chyba u zpracovani nahravky cca cislo. 2900 neco a souvisejici collate_fn
- vysledky dosavadniho trenovani: zatim dost nanic
- loss funkce pro moznost sledovat vizualne progress trenovani
- zobrazeni rekonstruovanych nahravek v programu Audacity
- Dalsi kroky:
    - extrahovat jednotlive tridy do vlastnich souboru a parametrizovat instanciaci pro snazsi volani site v ruznych konfiguracich
    - [done] **Error:** zpustit sit na vsech nahravkach !bez trenovani!, abych mohl jednoduse opravit tu chybu se zpracovanim nahravek kolem c.2900
    - **Batch:** size batch > 1 -definovat collate_fn funkci
    - **zero padding** uplne na zacatku trenovani - jeste pred ResBlocky, protoze pokud budu mit vetsi batche, tak nahravky budou ruzne dlouhe, a to by zpusobovalo problemy s rozmery jako nyni, takze je potrebe je zarovnat na delku nejdelsi nahravky v batchi.
    - [done] **loss funkce:** ukladat si hodnoty loss nekam do souboru s moznosti to kdykoli vykreslit a pro perzistentnost techto dat. Ulozit tam i aktualni konfiguraci site (mozna do nazvu).
    - vypsat vsechny rozmery tensoru skrze sit v prubehu zpracovani a trenovani nahravek a zjistit, jestli tvary odpovidaji tem ve studii a jestli se vsude deje to, co se dit ma.
    - viz. ../konzultace_poznamky.txt

### ct 7. 10. 2019 - Konzultace
- chyba s indexem byla opravena a sit se zacala konecne ucit spravne separovat mluvci jiz po 3. epose
- sit projela vsech 20k ucicich nahravek
- dalsi kroky:
    - zprovoznit inferenci, pripadne moznost siti predat ID nahravky a nechat si vyhodit separovanou smes.
    - zkontrolovat rozmery v prubehu site


