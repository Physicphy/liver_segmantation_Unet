#!/bin/sh

set -e # exit if any command fail
# Colors
red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

OUTPUT_DIR="./outputs"
if [ -d "$OUTPUT_DIR" ]; then
  echo ">>>${red} ${OUTPUT_DIR} folder already exist... ${reset}"
else
  echo "${green}>>> Add outputs folder${reset}"
  mkdir ${OUTPUT_DIR}
fi

echo ""
DATASET_DIR="./dataset"
if [ -d "$DATASET_DIR" ]; then
  echo ">>>${red} ${DATASET_DIR} folder already exist... ${reset}"
else
  echo "${green}>>> Add ${DATASET_DIR} folder and subfolders${reset}"
  mkdir ${DATASET_DIR}
  mkdir "./dataset/train"
  mkdir "./dataset/train/masks"
  mkdir "./dataset/train/originals"

  mkdir "./dataset/test"
  mkdir "./dataset/test/masks"
  mkdir "./dataset/test/originals"
fi

echo ""
echo "${green}>>> Creating conda enviroment...${reset}"

condaEnvName="liverSegmentationUnet"
echo "${green}>>> Default name: ${condaEnvName} ${reset}"

#while true; do
#    read -p "${green}        >>> Do you wish to keep default conda enviroment name (y/n)? ${reset}" yn
#    case $yn in
#        [Yy]* )
#	    conda create -n ${condaEnvName}
#	    conda activate ${condaEnvName}
#	    conda env update --file requirements.yml
#	    break
#	;;
#        [Nn]* )
#	    read -p "${green}        	>>> Type the name of conda enviroment you want: ${reset}" venvUser
#	    conda create -n ${venvUser}
#	    conda activate ${venvUser}
#	    conda env update --file requirements.yml
#	    exit
#	;;
#        * )
#	    echo "${red}        	>>> Please answer y or n ${reset}"
#	;;
#    esac
#done


