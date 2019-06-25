#!/usr/bin/env bash

# needs following 3 arguments to run the script:
# $1 : environment name
# $2 : belief_loss_type {task_agnostic, task_aware}
# $3 : belief_regularization {True, False}

function usage {
    echo '[Usage:] bash run_mujoco.sh [env] [belief_loss_type] [belief_regularization]'
}

if [ $# -ne 3 ]; then
    echo 'Illegal number of arguments'
    usage
    exit
fi

envs_list="Swimmer-v2 Hopper-v2 Walker2d-v2 InvertedPendulum-v2 InvertedDoublePendulum-v2 HalfCheetah-v2 Ant-v2 Humanoid-v2"

function list_include_env {
  local list="$1"
  local item="$2"
  if [[ $list =~ (^|[[:space:]])"$item"($|[[:space:]]) ]] ; then
    :
  else
    echo 'Unknown environment '$2
    exit
  fi
}

list_include_env "$envs_list" $1

python ./code/main.py -p with \
    environment.name=$1 \
    environment.config_path=$PWD'/code/conf/envParams.yaml' \
    algorithm.belief_loss_type=$2 \
    algorithm.belief_regularization=$3
