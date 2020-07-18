# TasNet
Bakalarska prace (Bachelor thesis) - Speaker separation in time domain.

## Studie
- [TasNet - v1](https://arxiv.org/abs/1809.07454v1) (Puvodni)
- [TasNet - v2](https://arxiv.org/abs/1809.07454v2)
- [TasNet - v3](https://arxiv.org/abs/1809.07454) (Nova prace TasNet - 2019 (v3) - zlepseny popis modelu atd.)

## Ruzne kurzy o NN, SI-SNR, examples
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
    - [DONE] **Error:** zpustit sit na vsech nahravkach !bez trenovani!, abych mohl jednoduse opravit tu chybu se zpracovanim nahravek kolem c.2900
    - [DONE] **Batch:** size batch > 1 -definovat collate_fn funkci
    - **zero padding** uplne na zacatku trenovani - jeste pred ResBlocky, protoze pokud budu mit vetsi batche, tak nahravky budou ruzne dlouhe, a to by zpusobovalo problemy s rozmery jako nyni, takze je potrebe je zarovnat na delku nejdelsi nahravky v batchi.
    - [DONE] **loss funkce:** ukladat si hodnoty loss nekam do souboru s moznosti to kdykoli vykreslit a pro perzistentnost techto dat. Ulozit tam i aktualni konfiguraci site (mozna do nazvu).
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
    - [TODO] vykreslit i validacni loss
    - [TODO] ukladat a upravit funkci aby vykreslila obe dve z trenovani i validacni a jinou barvou pres sebe
    - [DONE] nechat zpracovavat celou validacni mnozinu
    - nechat poradne natrenovat jednu sit s nejakou konfiguraci
    - [DONE] rozjet si-snr na testovaci mnozine a zprumerovat.
    - [TODO] make repo public a umoznit tak jednoduse ukladat/loadovat zdrojaky z/na git

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
    - [DONE] mini-batche
    - [DONE] segmentovat nahravky na 4s

### ut 3. 12. 2019 - Konzultace
- skipped

### pa 13. 12. 2019 - Konzultace
- pro trenovani zkusit nahravky rozdelit po  4 sekundach. kratsi dopaddovat nulama, delsi rozdelit a treba pro nahravku 1-6 vzit 1-4 a pak 3-6, aby se prolinaly konce/zacatky.
- pozn.: pvni dimenze je obvykle velikost batche / v reshapech atd. ta by mela zustat vzdy na prvnim miste
- pozn.: loss funkci bych mel spocitat pro s1 a s2 zvlast
    - +pred loss pak zase vyhodit to padovani ??
- dalsi kroky:
    - [DONE] je nutne udelat Cross-validaci viz papir a ruzne poscitat vysledky loss funkce a vybrat nejlepsi a tu pouzit pro backprop.
    - [DONE] segmentace a padding a minibatche

### ut 14. 1. 2020 - Konzultace
- **Prezentace**
    - Obsah: https://wis.fit.vutbr.cz/FIT/st/bci-com-l.php.cs?id=9
    - Do kdy odevzdat zdrojaky:  neni nutne
    - Co vse odevzdat: neni nutne
- upgrade RAMky
- segmentace nahravek - v main.py ukazat
- zprovoznena segmentace a cross validace mezi loss hodnotami
- DOTAZ: jak je to se segmentaci u Validace a Test?
    - u Validacni jo
    - u Testovaci ne (cele nahravky - pripadne batch o velikosti 1)
- DOTAZ: ma byt nekde minus u siSNR obj funkce? -sisnr() - sisnr() a nechat min.
- DOTAZ: delka segmentu a delka prekryti: segmenty nechat a prekryti oddelat.
- Error - index out of range - v main v cyklu segmentovat cely dataset a najit chybu.
- dalsi kroky:
    - [DONE] oddelat prekryti
    - [DONE] batch size tak velkou co to jde ... az dokud to nespadne pametove
    - [DONE]mozna neni nutne paddovani nulama, ale u posledniho nekompletniho segmentu
    vzit ty 4 sekundy od konce
    - [DONE] cca do ctvrtka nastin prezentace
    - [DONE] U testovaci sady zadne batche, ale cele nahravky (prip. batch o velikosti 1)

### pa 17. 1. 2020 - Konzultace
- konzultace k prezentaci

### ut 21.1.2020 - Obhajoba
- **Feedback** viz dokument

### po 9.3.2020
    [DONE] Zmenena segmentace tak, aby vzala pro posledni segment posledni 4 sekundy, misto paddovani nulama. Nulama se paduje uz jen u nahravek, ktere jsou kratsi nez 4 sekundy.
    [DONE] zkontrolovat pri inferenci pocet kanalu a vzorkovani.
    [DONE] Zprovozneno Testovani
    [DONE] Zprovoznena inference
    [DONE] Sit se trenuje
    [DONE] je potreba spravit trenovani. Spravne se projede jen prvni epocha, a z nejakeho duvodu se to dal .. - kazda druha epocha se neprovede, ostatni jsou moc rychle.
    - Pametova narocnost pri inferenci nahravky dlouhe 7minut (350MB)
    [DONE] nemela by se loss pocitat jako average?
    [DONE] scripty pro gdrive
    [DONE] proc loss klesa do minusu

### 14. 4. 2020 - Online konzultace
    - veci kolem textu a experimentu

### 20. 4. 2020 - Online konzultace
    - veci kolem textu a experimentu

### 2. 6. 2020 - Online konzultace
    - konzultace textu
    - oznaceni prevzatych funkci ve zdrojovem kodu
    - komentare k textu
    - pohrat si s Learning rate, pouzit tu co dava nejlepsi vysledky. Zvolit nejakou idealni pro vsechny velikosti site.
    - Pozn: Vetsi model neznamena lepsi loss

### 9. 6. 2020 - Online konzultace
    - konzultace k textu:
    - info o kapitole TasNet
        - prvne rozvrhnout podle sebe, pak az kouknout na studii
        - nejde to vymyslet cele znova
        - pri parafrazi nejakeho bloku ocitovat i citace z toho bloku a procist si letmo co cituji
    - uprava algoritmuuu
        - najit si konvence a upravit podle toho
        - udelat komentare treba kurzivou, zbytek jinak
    - obrazek se segmentaci a minibatchema - mozna do nej zakomponovat alespon 2 nebo 3 nahravky zasebou, ale aby to nebylo zmatene
    - dopsat sisnr a loss podkapitolu v implementaci, a pokud by se pak ukazalo, ze bude lepsi to hodit do TasNet kapitoly, tak to presunout.
    - **moznost osobni konzultace**

### 16. 6. 2020 - Online konzultace
    - Done:
        - upraven Uvod a ref na studii
        - upraven text podle komentaru
        - ozdrojovane nejake obrazky kterymi jsem se inspiroval
        - Obj funkce, mseloss, cross entrophy

### 3. 7. 2020 - face to face konzultace
    - Prokonzultovat SDR, STOI, PESQ vysledky testovani
        - impl PESQ vypocet - min, nebo aktualni max?
    - natrenovane modely info
    - zkratky a anglicke nazvy metod... kdy bold, kdy italic 
    - tucnym setrit, jen kdyz neco hodne zvyraznit, ale moc nepouzivat.
    - kurziva taky zridka. proste podle citu
    - TODO:
    - [] ukazat komentar prevzateho kodu - jestli to staci.
    - [DONE] do nazvu podkapitol nedavat zkratku, jen ten angl nazev - relu ...
    - [DONE] feedforward -> dopredna
    - [DONE] v rovnicich vektory a matice jsou tucne.
    - v pondeli a utery napsat experimenty, pres zbytek tydne doupravit text podle poznamek vedouci a dalsi pondeli odevzdat.
    - ! upravit italic a bold text
    - ! upravit algoritmy
    - ! dopsat validacni dataset v trenovani
    - ! texttt pro zdrojove veci 
