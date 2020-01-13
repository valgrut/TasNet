# TasNet
Bakalarska prace (Bachelor thesis) - Speaker separation in time domain.

## Resources k praci
- [TasNet - v1](https://arxiv.org/abs/1809.07454v1) (Puvodni)
- [TasNet - v2](https://arxiv.org/abs/1809.07454v2)
- [TasNet - v3](https://arxiv.org/abs/1809.07454) (Nova prace TasNet - 2019 (v3) - zlepseny popis modelu atd.)
- [SI-SNR source code repo](https://github.com/craffel/mir_eval/blob/master/mir_eval/separation.py)
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
- [Audacity]()

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
    - [DONE] extrahovat jednotlive tridy do vlastnich souboru a parametrizovat instanciaci pro snazsi volani site v ruznych konfiguracich
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

### ut 12. 11. 2019 - Konzultace
- ukazka vykresleni loss funkce
- ukazka moznosti inference
- dalsi kroky:
    - [DONE] Pridat zde do testovani SI_SNR  a pro kazdou nahravku a vysledky zprumerovat a vyhodnotit
    - [DONE] Rozdlit do souboru a parametrizovat tridy a presunout je do zvlastnich souboru .py
    - vykreslit i validacni loss
    - ukladat a upravit funkci aby vykreslila obe dve z trenovani i validacni a jinou barvou pres sebe
    - [done] nechat zpracovavat celou validacni mnozinu
    - nechat poradne natrenovat jednu sit s nejakou konfiguraci
    - [DONE] rozjet si-snr na testovaci mnozine a zprumerovat.
    - make repo public a umoznit tak jednoduse ukladat/loadovat zdrojaky z/na git

### ut 19. 11. 2019 - Konzultace
- skipped

### ut 26. 11. 2019 - Konzultace
- **[INFO] 21. 1. 2020 Obhajoba - 7min, prezentace co delam / udelal/ co funguje, nefunguje...**
- Nalezen problem s nacitanim checkpointu, ze nesedi jaksi rozmery, i kdyz by sedet mely.
    - je mozne ze to je kvuli nejake zmene v siti a checkpoint tuto zmenu nereflekoval
- dalsi kroky:
    - [TODO] Zkusit udelat novy checkpoint s ruznymi X a R a nasledne ho zkusit nahrat.
    - [TODO] kouknout na pytorch errory, co mi to hazi uz nejakou dobu, mela by to byt hovadina jen
    - [DONE] napsat si svou loss funkci - SI-SNR jako learning objective.
    - [TODO] mini-batche
    - segmentovat nahravky na 4s - problem: nahravky jsou prumerne 4s dlouhe, takze nevim, jestli to neni zbytecne.

### ut 3. 12. 2019 - Konzultace
- skipped

### pa 13. 12. 2019 - Konzultace
- 
