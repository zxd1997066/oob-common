#!/bin/bash
set -xe

# WORKSPACE
if [ "${WORKSPACE}" == "" ];then
    WORKSPACE="${PWD}/logs"
fi
# extra env
if [ "${OOB_ADDITION_ENV}" != "" ];then
    echo ${OOB_ADDITION_ENV} > env.tmp
    sed -i 's/,/\n/g' env.tmp
    sed -i 's/=/="/;s/$/"/' env.tmp
    sed -i 's/^/export /' env.tmp
    sed -i 's/|/ /g' env.tmp
    cat env.tmp && source env.tmp && rm -f env.tmp
fi

# env
function set_environment {
    # unicode
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export DEBIAN_FRONTEND=noninteractive
    # update numpy
    pip install -U numpy==1.23.5
    # basic env for CPU
    if [ "${device}" != "cuda" ];then
        # cpu envs
        export KMP_BLOCKTIME=1
        export KMP_AFFINITY=granularity=fine,compact,1,0
        if [ "${framework}" == "pytorch" ];then
            # intel OMP + Jemalloc for pytorch
            pip install psutil
            if [ $(conda > /dev/null 2>&1 && echo $? ||echo $?) -eq 0 ];then
                export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
                export LIBRARY_PATH=${LIBRARY_PATH}:${CONDA_PREFIX}/lib
                export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
                export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
            else
                export LD_PRELOAD=$(find /usr -name "libjemalloc.so" |head -1)
                export LD_PRELOAD=$(find /usr -name "libiomp5.so" |head -1)
            fi
            export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
        fi
        if [ "${framework}" == "tensorflow" ];then
            # tensorflow
            export TF_ENABLE_ONEDNN_OPTS=1
            # export TF_ENABLE_MKL_NATIVE_FORMAT=1
            export TF_ONEDNN_THREADPOOL_USE_CALLER_THREAD=true
            export TF_ONEDNN_THREAD_PINNING_MODE=none
        fi
    else
        # cuda env
        export NVIDIA_TF32_OVERRIDE=0
        if [ "${framework}" == "tensorflow" ];then
            # XLA off for tensorflow
            export TF_XLA_FLAGS="--tf_xla_auto_jit=-1"
            # export TF_GPU_ALLOCATOR=cuda_malloc_async
        fi
    fi
}

# logs
function collect_perf_logs {
    # latency
    latency=($(grep -i 'inference latency:' ${log_dir}/rcpi* |sed -e 's/.*atency://;s/[^0-9.]//g' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    # throughput
    throughput=($(grep -i 'inference Throughput:' ${log_dir}/rcpi* |sed -e 's/.*hroughput://;s/[^0-9.]//g' |awk  -v i=$instance '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num == 0) {
                num = i;
            }
            printf("%d  %.6f", num, sum);
        }
    '))
    # summary
    if [ "$BUILD_URL" != "" ];then
        link="${BUILD_URL}artifact/$(basename ${log_dir})"
    else
        link="${log_dir}"
    fi
    printf "${framework},${model_name},${mode_name},${precision},${batch_size}," |tee -a ${WORKSPACE}/summary.log
    printf "${cores_per_instance},${throughput[0]},${throughput[1]},${link} ,${device},${latency}\n" |tee -a ${WORKSPACE}/summary.log
    set +x
    mv timeline/ ${log_dir}/ || true
    echo -e "\n\n-------- Summary --------"
    sed -n '1p;$p' ${WORKSPACE}/summary.log |column -t -s ','
}

# device info
function fetch_device_info {
    # hardware info
    hostname
    cat /etc/os-release || true
    cat /proc/sys/kernel/numa_balancing || true
    # scaling_governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    # if [ $(sudo -n true > /dev/null 2>&1 && echo $? || echo $?) -eq 0 ];then
    #     if [ "${scaling_governor}" != "performance" ];then
    #         # set frequency governor to performance mode
    #         sudo cpupower frequency-set -g performance || true
    #     fi
    #     # clean cache
    #     sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
    # else
    #     echo "[INFO] You do NOT have ROOT permission to set system config."
    #     echo "       The frequency governor is ${scaling_governor}."
    # fi
    lscpu
    uname -a
    free -h
    numactl -H
    sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
    cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
    phsical_cores_num=$(echo |\
            awk -v sockets_num=${sockets_num} -v cores_per_socket=${cores_per_socket} '{
        print sockets_num * cores_per_socket;
    }')
    numa_nodes_num=$(numactl -H |grep 'node [0-9]* cpus: [0-9].*' |wc -l)
    threads_per_core=$(lscpu |grep 'Thread(s) per core:' |sed 's/[^0-9]//g')
    cores_per_node=$(numactl -H |grep "node 0 cpus:" |sed 's/.*://' |awk -v tpc=$threads_per_core '{print int(NF / tpc)}')
    if [ "${OOB_HBM_FLAT}" != "" ];then
        hbm_index=$numa_nodes_num
    else
        hbm_index=0
    fi
    # cores to use
    if [ "${cores_per_instance,,}" == "1s" ];then
        cores_per_instance=${cores_per_socket}
    elif [ "${cores_per_instance,,}" == "1n" ];then
        cores_per_instance=${cores_per_node}
    fi
    # cpu model name
    cpu_model="$(lscpu |grep 'Model name:' |sed 's/.*: *//')"
    if [[ "${cpu_model}" == *"8180"* ]];then
        device_type="SKX"
    elif [[ "${cpu_model}" == *"8280"* ]];then
        device_type="CLX"
    elif [[ "${cpu_model}" == *"8380H"* ]];then
        device_type="CPX"
    elif [[ "${cpu_model}" == *"8380"* ]];then
        device_type="ICX"
    elif [[ "${cpu_model}" == *"AMD EPYC 7763"* ]];then
        device_type="MILAN"
    else
        device_type="Unknown"
    fi
    # cpu array
    if [ "${numa_nodes_use}" == "all" ];then
        numa_nodes_use_='1,$'
    elif [ "${numa_nodes_use,,}" == "1s" ];then
        numa_nodes_use_="1,$[ $cores_per_instance / $cores_per_node ]"
    elif [ "${numa_nodes_use}" == "0" ];then
        numa_nodes_use_=1
    else
        numa_nodes_use_=${numa_nodes_use}
    fi
    if [ "${device}" != "cuda" ];then
        device_array=($(numactl -H |grep "node [0-9]* cpus:" |sed "s/.*node//;s/cpus://" |sed -n "${numa_nodes_use_}p" |\
                awk -v cpn=${cores_per_node} '{for(i=1;i<=cpn+1;i++) {printf(" %s ",$i)} printf("\n");}' |grep '[0-9]' |\
                awk -v cpi=${cores_per_instance} -v cpn=${cores_per_node} -v cores=${OOB_TOTAL_CORES_USE} -v hi=${hbm_index} '{
            if(cores == "") { if(cpi > cpn) {cores = cpi}else {cores = NF} }
            for( i=2; i<=cores; i++ ) {
                if($i != "") {
                    if((i-1) % cpi == 0) {
                        print $i";"$1+hi
                    }else {
                        printf $i","
                    }
                }
            }
        }' |sed "s/,$//"))

        export OMP_NUM_THREADS=$(echo ${device_array[0]} |awk -F, '{printf("%d", NF)}')
    else
        if [ $(nvidia-smi -L |grep 'MIG' |wc -l) -ne 0 ];then
            device_array=($(nvidia-smi -L |grep 'MIG' |sed 's/.*UUID: *//;s/).*//' |sed -n "${numa_nodes_use_}p"))
        else
            device_array=($(nvidia-smi -L |grep 'NVIDIA' |sed 's/.*UUID: *//;s/).*//' |sed -n "${numa_nodes_use_}p"))
        fi
	export CUDA_VISIBLE_DEVICES=${device_array[0]}
    fi
    instance=${#device_array[@]}

    # environment
    gcc -v
    python -V
    conda list || echo
    pip list
    fremework_version="$(pip list |& grep -E "^torch[[:space:]]|^pytorch[[:space:]]" |awk '{printf("%s",$2)}')"
    printenv
}

function logs_path_clean {
    # logs saved
    log_dir="${device}-${framework}-${model_name}-${mode_name}-${precision}-bs${batch_size}-"
    log_dir+="cpi${cores_per_instance}-ins${instance}-nnu${numa_nodes_use}-$(date +'%s')"
    log_dir="${WORKSPACE}/$(echo ${log_dir} |sed 's+[^a-zA-Z0-9./-]+-+g')"
    mkdir -p ${log_dir}
    if [ ! -e ${WORKSPACE}/summary.log ];then
        printf "framework,model_name,mode_name,precision,batch_size," | tee ${WORKSPACE}/summary.log
        printf "cores_per_instance,instance,throughput,link ,device,latency\n" | tee -a ${WORKSPACE}/summary.log
    fi
    # exec cmd
    excute_cmd_file="${log_dir}/${framework}-run-$(date +'%s').sh"
    rm -f ${excute_cmd_file}
    rm -rf ./timeline
}

function init_params {
    device='cpu'
    framework='framework'
    model_name='model_name'
    mode_name='realtime'
    precision='float32'
    batch_size=1
    numa_nodes_use='all'
    cores_per_instance=4
    num_warmup=20
    num_iter=200
    profile=0
    dnnl_verbose=0
    channels_last=1
    # addtion args for exec
    addtion_options=" ${OOB_ADDITION_PARAMS} "
    #
    for var in $@
    do
        case ${var} in
            --device=*)
                device=$(echo $var |cut -f2 -d=)
            ;;
            --framework=*)
                framework=$(echo $var |cut -f2 -d=)
            ;;
            --model_name=*)
                model_name=$(echo $var |cut -f2 -d=)
            ;;
            --mode_name=*)
                mode_name=$(echo $var |cut -f2 -d=)
            ;;
            --precision=*)
                precision=$(echo $var |cut -f2 -d=)
            ;;
            --batch_size=*)
                batch_size=$(echo $var |cut -f2 -d=)
            ;;
            --numa_nodes_use=*)
                numa_nodes_use=$(echo $var |cut -f2 -d=)
            ;;
            --cores_per_instance=*)
                cores_per_instance=$(echo $var |cut -f2 -d=)
            ;;
            --num_warmup=*)
                num_warmup=$(echo $var |cut -f2 -d=)
            ;;
            --num_iter=*)
                num_iter=$(echo $var |cut -f2 -d=)
            ;;
            --profile=*)
                profile=$(echo $var |cut -f2 -d=)
            ;;
            --dnnl_verbose=*)
                dnnl_verbose=$(echo $var |cut -f2 -d=)
            ;;
            --channels_last=*)
                channels_last=$(echo $var |cut -f2 -d=)
            ;;
            *)
                addtion_options+=" $var "
            ;;
        esac
    done
    # Profile
    if [ "${profile}" == "1" ];then
        addtion_options+=" --profile "
    elif [ "${profile}" != "" ] && [ "${profile}" != "0" ];then
        addtion_options+=" --profile $profile "
    fi
    # DNN Verbose
    if [ "${dnnl_verbose}" != "" ] && [ "${dnnl_verbose}" != "0" ];then
        export MKLDNN_VERBOSE=$dnnl_verbose
        export DNNL_VERBOSE=$dnnl_verbose
        export ONEDNN_VERBOSE=$dnnl_verbose
    else
        unset DNNL_VERBOSE MKLDNN_VERBOSE ONEDNN_VERBOSE
    fi
}
