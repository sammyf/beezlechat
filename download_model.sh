#!/bin/bash
echo Model Path?
read MODEL
echo "Revision?"
echo "(enter 'default' for Default model, otherwise defaulting to 'gptq-4bit-32g-actorder_True')?"
read TAG
REV="--revision gptq-4bit-32g-actorder_True"
if [ $TAG == "default" ]; then
  REV=""
elif [ $TAG != "" ]; then
  REV="--revision $TAG"
fi
DEST=$(echo "$MODEL" | sed "s/\//_/g" )
echo $DEST
mkdir "models/$DEST"
echo "Downloading $MODEL $REV to models/$DEST"
echo huggingface-cli download $MODEL $REV --local-dir "models/$DEST" --local-dir-use-symlinks False
source bchat/bin/activate && huggingface-cli download $MODEL $REV --local-dir "models/$DEST" --local-dir-use-symlinks False
