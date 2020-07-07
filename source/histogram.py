#! /usr/bin/env python3

## - script precte soubor s hodnotami sdr pro jednotlive promluvy (nazev, sdr)
## - ty bude ukladat do mapy (hodnota, pocet)
## - vykresli na konci graf hodnot

import math
import numpy as np
import argparse
import matplotlib.pyplot as plt

def parseGendre(mixname):
    if mixname.find("MM") > -1:
        return "MM"
    elif mixname.find("MZ") > -1:
        return "MZ"
    elif mixname.find("ZM") > -1:
        return "MZ"
    else:
        return "ZZ"

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Choose the type of histogram')

    parser.add_argument('--sdr-file-path',
            dest='sdr_file_path',
            type=str,
            help='Full path to file with sdr values')

    parser.add_argument('--round',
            dest='round',
            default=4,
            type=int,
            help='Value to which round sdr values. Default value = 4')

    parser.add_argument('--aggregate',
            dest='aggregation',
            default='normal',
            type=str,
            help='Type of aggregation of sdr. Allowed values: genre, normal')

    args = parser.parse_args()
    print(args)

    # parameter check
    if not args.sdr_file_path:
        print("Chyba: Pozadovana cesta k souboru s hodnotami sdr")
        exit(1)

    if args.round < 1 or args.round > 20:
        print("Chyba: Parametr round prijima pouze hodnotu v rozmezi 1 - 20")
        exit(2)

    if args.aggregation not in ["genre", "normal"]:
        print("Chyba: Povolene parametry jsou poze genre a normal")
        exit(3)

    # Read the file line by line and split the values and save them into the map
    # (Init map with sdr values)
    speech_sdr_map = {}
    filepath = args.sdr_file_path
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            line = line.split('\n')
            line = line[0].split(' ')
            sdr = np.array(line[1]).astype(np.float64)
            speech_sdr_map.update({line[0]+str(cnt) : sdr})
            # print("Line {}: {}".format(line[0]+str(cnt), line[1]))
    
    # Process data - create histogram of sdr values
    modus = 0
    median = 0
    average = 0
    histogram = {}
    histogram_gendre = {}
    for mixture, sdr in speech_sdr_map.items():
        round_sdr = np.round(sdr, args.round)
        # print(mixture, " : ", round_sdr)

        # Create histogram of sdr
        if round_sdr not in histogram:
            histogram.update({round_sdr : 1})
        else:
            histogram[round_sdr] += 1

        # Create histogram of sdr based on genre
        combination = parseGendre(mixture)
        if combination not in histogram_gendre:
            histogram_gendre.update({combination : []})
        else:
            histogram_gendre[combination].append(round_sdr)
    
    # Calculate mean... for gendre histogram
    gendres = ["MM", "MZ", "ZZ"]
    MM_avg = sum(histogram_gendre["MM"]) / len(histogram_gendre["MM"])
    MZ_avg = sum(histogram_gendre["MZ"]) / len(histogram_gendre["MZ"])
    ZZ_avg = sum(histogram_gendre["ZZ"]) / len(histogram_gendre["ZZ"])
    gendre_sdr = [MM_avg, MZ_avg, ZZ_avg]

    # Print histogram values
    sort_key = 0            # index of array to be used for sorting
    reverse_sort = True    # reverse sort - asc / desc
    sorted_histogram = sorted(histogram.items(), key=lambda x: x[sort_key], reverse=reverse_sort)
    # for sdr, count in sorted_histogram.items():
    x = [] 
    y = []
    for pair in sorted_histogram:
        # print(pair[0], " : ", pair[1])
        x.append(pair[0])
        y.append(pair[1])
        
    # Draw histogram
    plt.title('SDR Histogram')
    plt.xlabel('SDR value')
    plt.ylabel('Number of SDR occurence')
    plt.bar(x, y, width=0.1, color='g')
    plt.show()
    
    plt.title('SDR Histogram')
    plt.xlabel('Pohlavi mluvcich na smesi')
    plt.ylabel('Prumerne SDR')
    plt.bar(gendres, gendre_sdr, width=0.1, color='g')
    plt.show()




