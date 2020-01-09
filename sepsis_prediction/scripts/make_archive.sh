#!/usr/bin/env bash

mkdir team6_sepsis_prediction

target="team6_sepsis_prediction"

mkdir $target

cp doc/*.{pdf,pptx} $target
rsync -Rr doc $target
rsync -Rr data/data20191125.csv $target
rsync -Rr data/text_features_1575527882.csv $target
rsync -Rr output $target
rsync -Rr scripts $target
rsync -Rr src $target
rsync -Rr tst $target
rsync -Rr environment.yml $target
rsync -Rr README.md $target

tar -czvf ${target}.tar.gz $target
rm -rf $target

