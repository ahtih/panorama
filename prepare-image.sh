#!/bin/sh

DESTDIR=viewable-images

id=`echo $1 | tr -d --complement '0123456789' | sed -e 's/^0*//' `
orig_size=`identify $1 | head -1 | cut -d' ' -f3 | cut -dx -f1`

mkdir $DESTDIR/$id

for quality_code in 11 12 13 14 15 ; do
	target_size=$(( 1 << $quality_code ))

	if [ $target_size -lt $orig_size ] ; then
		echo $target_size 'resize'
		convert $1 -resize $target_size $DESTDIR/$id/${quality_code}.jpg
	else
		echo $target_size 'copy'
		cp $1 $DESTDIR/$id/${quality_code}.jpg
		break
	fi

	done
