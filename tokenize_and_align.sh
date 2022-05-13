# Shell script to get the tokens and alignments data, courtesy of Steven Cao (shcao@stanford.edu)
# Tools used:
# https://github.com/clab/fast_align for word alignment
# https://github.com/moses-smt/mosesdecoder for tokenization

LANG=(de fr bg el es)
NUM_LANG=${#LANG[@]}

for (( i=0; i<${NUM_LANG}; i++ ));
do
	echo ${LANG[$i]}
	
	# Tokenize
	mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${LANG[$i]} \
	  < europarl-v7.${LANG[$i]}-en.${LANG[$i]} > europarl-v7.${LANG[$i]}-en.${LANG[$i]}.token

  mosesdecoder/scripts/tokenizer/tokenizer.perl -l en \
	  < europarl-v7.${LANG[$i]}-en.en > europarl-v7.${LANG[$i]}-en.en.token
	
	# Combine into one file
	paste europarl-v7.${LANG[$i]}-en.${LANG[$i]}.token europarl-v7.${LANG[$i]}-en.en.token | sed 's/ *\t */ ||| /g' > europarl-v7.${LANG[$i]}-en.token
	
	# Remove (..), short/empty lines, soft dashes, lines with backslash, put special characters back
	sed 's/( .. ) //g' europarl-v7.${LANG[$i]}-en.token | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' | awk 'length($0)>100' | awk 'length($0)<2000' > europarl-v7.${LANG[$i]}-en.token.clean
	
	# Remove special spaces
	python clean_data.py europarl-v7.${LANG[$i]}-en.token.clean europarl-v7.${LANG[$i]}-en.token.cleantemp
	
	rm europarl-v7.${LANG[$i]}-en.token.clean
	
	mv europarl-v7.${LANG[$i]}-en.token.cleantemp europarl-v7.${LANG[$i]}-en.token.clean
	
	echo Running fast_align...
	# Run fast_align
	fast_align/build/fast_align -i europarl-v7.${LANG[$i]}-en.token.clean -d -o -v > europarl-v7.${LANG[$i]}-en.align
	
	fast_align/build/fast_align -i europarl-v7.${LANG[$i]}-en.token.clean -d -o -v -r > europarl-v7.${LANG[$i]}-en.reverse.align
	
	fast_align/build/atools -i europarl-v7.${LANG[$i]}-en.align -j europarl-v7.${LANG[$i]}-en.reverse.align -c intersect > europarl-v7.${LANG[$i]}-en.intersect
	
done
