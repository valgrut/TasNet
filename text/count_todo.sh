#! /usr/bin/env bash
HEAD=$(cat projekt-01-kapitoly-chapters.tex | nl | grep "Experimenty a vyhodnocení" | c1)
cat projekt-01-kapitoly-chapters.tex | head -n $HEAD | grep todo | wc -l
