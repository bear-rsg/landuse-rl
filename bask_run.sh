#!/bin/bash

# Copyright 2023 University of Birmingham, Thomas Kappas Simon Hartley, Adrian Garcia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

set -e 

# move to dir which this script is in: 
cd "${0%/*}"

export PROJECTROOT=$(pwd)

#colours for the terminal
export RED='\e[38;5;198m'
export RED_END='\e[0m'
export GOLDEN='\e[38;5;214m'
export ORANGE='\e[38;5;202m'
export NOCOLOR='\e[0m'

# Default values for command line values
odir="${PROJECTROOT}/Output/$(date +"%Y-%m-%dT%H-%M")"
save_freq=0 
venv_name="venv_landuse" 
episode_max=100
del_env='N'
data_dir="${PROJECTROOT}/apps/${venv_name}/lib64/python3.10/site-packages/demoland_engine/data/"    


#######################################
# Load Slurm modules from Baskerville 
# GLOBALS:
#   RED, RED_END, NOCOLOR
# ARGUMENTS:
#   None
# OUTPUTS:
#   Load Modules 
# RETURN: 
#######################################
function Bask_load_modules_landUse() {

    echo -e "${RED}1.1a load modules landUse:Start ${NOCOLOR}"
    module purge
    module load bask-apps/live
    module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
    echo -e "1.1a load modules landUse:done${RED}Loaded Modules${RED_END}${NOCOLOR}" 
}

#######################################
# create venv from baskerville loaded modules and requirements.txt
# GLOBALS:
#   PROJECTROOT, RED, RED_END, NOCOLOR
# ARGUMENTS:
#   None
# OUTPUTS:
#   Load Modules 
# RETURN: 
#######################################
function Bask_make_venv_landuse(){

    echo -e "1.1b ${RED}Bask_make_venv_landuse:${RED_END} move to apps folder"
    mkdir "${PROJECTROOT}/apps/"
    cd 	"${PROJECTROOT}/apps/" 

    echo -e "${RED}Bask_make_venv_landuse:${RED_END}${NOCOLOR}python -m venv ${venv_name} "   
    python -m venv "${venv_name}"
    echo -e "${RED}Bask_make_venv_landuse:${RED_END}${NOCOLOR}python -m venv ${venv_name} " 
    source "${PROJECTROOT}/apps/${venv_name}/bin/activate"
    echo -e "${RED}Bask_make_venv_landuse:${RED_END}${NOCOLOR}${PROJECTROOT}/apps/${venv_name}/bin/activate" 
    echo -e "${RED}make_venv_landuse:${RED_END}${NOCOLOR}source ${PROJECTROOT}/apps/${venv_name}/bin/activate "   
    pip install geopandas
    pip install -r ../requirements.txt
    pip install torchsummary 

    source "${PROJECTROOT}/apps/${venv_name}/bin/activate" deactivate
    echo -e "1.1b ${RED}Bask_make_venv_landuse:${RED_END} done"
}

#######################################
# test venv_landuse loaded py modules  
# GLOBALS:
#   PROJECTROOT, RED, RED_END, venv_name
# ARGUMENTS:
#   None
# OUTPUTS:
#   Load Modules 
# RETURN: 
#######################################
function test_venv_landuse(){
    source "${PROJECTROOT}/apps/${venv_name}/bin/activate"
    echo -e "1.1c ${RED}test_venv_landuse:${RED_END}"
    echo -e "test_sklearn version:"
    python -c "import sklearn; print(sklearn.__version__)"
    echo -e "test pandas version:"
    python -c "import pandas; print(pandas.__version__)"
    echo -e "test libpysal version:"
    python -c "import libpysal; print(libpysal.__version__)"
    echo -e "test xarray version:"
    python -c "import xarray; print(xarray.__version__)"
    echo -e "test pyarrow version:"
    python -c "import pyarrow; print(pyarrow.__version__)"
    echo -e "test joblib version:"
    python -c "import joblib; print(joblib.__version__)"
    echo -e "test demoland_engine version:"
    python -c "import demoland_engine; print(demoland_engine.__version__) "
    source "${PROJECTROOT}/apps/${venv_name}/bin/activate" deactivate
    echo -e "1.1c ${RED}test_venv_landuse:${RED_END} done"
}

#######################################
# clean the environment
# GLOBALS:
#   PROJECTROOT 
# ARGUMENTS:
#   None
# OUTPUTS:
#   None 
# RETURN: 
####################################### 
function clean(){
    echo -e "Clean up build files"
    rm -rf "${PROJECTROOT}/apps/"
    echo "clean Output?"
}

#######################################
# usage
# GLOBALS:
#   PROJECTROOT,venv_name
# ARGUMENTS:
#   None
# OUTPUTS:
#   None 
# RETURN: 
####################################### 
usage() {
  echo "USAGE"
  echo "  $(basename $0) -o dir [-s save_freq] [-e venv_environment]"
  echo
  echo "DESCRIPTION"
  echo "  Create lamd use env if needed and train for p episode max model saving weights every s"
  echo 
  echo "OPTIONS                                    [DEFAULT]" 
  echo "  -o: Output directory                     [${PROJECTROOT}/Output] "
  echo "  -s:  iterations to auto save             [0- no saveing]"
  echo "  -e: virtual environment name             [${venv_name}]"
  echo "  -p: number episodes max to train         [${episode_max}]"
  echo "  -d: delete env after use                 [N]"
}

#######################################
# exit abnormaly
# GLOBALS: 
# ARGUMENTS:
#   None
# OUTPUTS:
#   usage text
# RETURN: 
####################################### 
function exit_abnormal() {                         # Function: Exit with error.
  usage 
  exit 1
}

#######################################
# call_python
# GLOBALS: PROJECTROOT, odir, data_dir, save_freq
# ARGUMENTS:
#   None
# OUTPUTS:
#   Python outputs
# RETURN: 
####################################### 
function call_python() {                          
    python "${PROJECTROOT}/main.py" -vv --result_dir "${odir}" --data_dir "${data_dir}" --save_freq "${save_freq}"
}

#######################################
# main
# GLOBALS:
#   PROJECTROOT,odir,venv_name,checkpoints,episode_max,del_env
# ARGUMENTS:
#   None
# OUTPUTS:
# RETURN: 
####################################### 
function main() {

    echo "odir->"${odir}
    echo "venv_name->"${venv_name}
    echo "save_freq->"${save_freq}
    echo "episode_max->"${episode_max}
    echo "del_env->"${del_env}
    echo "data_dir->"${data_dir} 
    # make the output folder
    if [[ ! -e $odir ]]; then
      mkdir -p ${odir}
    elif [[ ! -d $odir ]]; then
      echo "$odir already exists but is not a directory"
      exit_abnormal
    fi

    
    start=`date -u +%s`
    Bask_load_modules_landUse
    ACTFILE="${PROJECTROOT}/apps/${venv_name}/bin/activate"
    if test -f "$ACTFILE"; then
      echo "$ACTFILE exists."
    else
      Bask_make_venv_landuse
    fi
    
    test_venv_landuse
    
    #***********************************************************************
    # Do LandUse training and work
    call_python
    #***********************************************************************
    if [[ $del_env =~ ^[Yy]$ ]]
        then
        # do rm of env folder
        clean
    fi; 
    end=`date -u +%s`

    runtime=$(date -u -d "0 $end seconds - $start seconds" +"%H:%M:%S")
  
    echo -e "${RED}main:Total${RED_END} Time Started:\e[0m $(date -d @${start})"
    echo -e "${RED}main:${RED_END}  Total Time Taken: $runtime"

}


#######################################
# get/ parse command line options
#######################################

[[ "$1" = -h ]] && { usage; exit; } 

while getopts ":o:s:e:p:d:" opt; do
  case $opt in
    o) odir=${OPTARG%/}; ;;  # Remove trailing slash if present
    e) venv_name=$OPTARG;      ;;    
    d) del_env=$OPTARG;      ;;  
    s) 
      save_freq=$OPTARG                     # Set $checkpoints to specified value.
      re_isanum='^[0-9]+$'                    # Regex: match whole numbers only
      if ! [[ $save_freq =~ $re_isanum ]] ; then   # if $checkpoints not whole:
        echo "Error: checkpoints must be a positive, whole number."
        exit_abnormal                     # Exit abnormally.
      fi;      ;; 
    p)  
      episode_max=$OPTARG                                     # Set $episode_max to specified value.
      re_isanum='^[0-9]+$'                                    # Regex: match whole numbers only
      if ! [[ $episode_max =~ $re_isanum ]] ; then            # if $episode_max not whole:
        echo "Error: episode_max must be a positive, whole number."
        exit_abnormal                                         # Exit abnormally.
      elif [ $episode_max -eq "0" ]; then                     # If it's zero:
        echo "Error: episode_max must be greater than zero."
      exit_abnormal                     # Exit abnormally.
      fi;      ;;
    \?) echo "Error: invalid option: -$OPTARG"; exit_abnormal          ;;
    :)  echo "Error: option -$OPTARG requires an argument"; exit 1 ;;
  esac
done
    



#######################################
# call main function to start 
main
#######################################

exit 0                                    # Exit normally.

 