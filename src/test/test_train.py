import argparse
import subprocess
import os
import sys
import platform

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


if __name__ == "__main__":
    from config.config_utils import parse_config
    from wickit.utils.log import log
    from wickit.utils.basic.string import dict_to_string
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("--config", help="trainer config file path")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--wait-to-start', type=str, default="")
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--multi', action='store_true', default=False)
    args = parser.parse_args()
    config_file = args.config

    program = 'src/test/test_trainer.py'
    config = parse_config(config_file, root_path="")
    num_gpu = config.trainer.num_gpu
    port = find_free_port()

    # os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_visible_devices']
    prefix = ""
    type_input = str(config.runtime.cuda_visible_devices)
    type_input = type_input.replace(" ", "").replace("(", "").replace(")", "").replace("\\", "")
    os.environ['CUDA_VISIBLE_DEVICES'] = type_input
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    # if platform.system() == "Windows":
    #     prefix = "set CUDA_VISIBLE_DEVICES={}; ".format(type_input)
    # elif platform.system() == "Linux":
    #     prefix = "CUDA_VISIBLE_DEVICES={}".format(type_input)
    train = args.train
    wait_to_start = args.wait_to_start
    test = args.test
    test_only = args.test_only
    if test_only:
        test = True
    resume = train and args.resume
    # multi = args.multi
    # debug_info = "export TORCH_DISTRIBUTED_DEBUG=INFO;"
    debug_info = ""
    # debug_info = "export TORCH_DISTRIBUTED_DEBUG=INFO; CUDA_LAUNCH_BLOCKING=1"
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    # os.environ['NCCL_DEBUG'] = 'INFO'
    command_str = None
    if train:
        config_str = "--config {} {} {} {}".format(config_file,
                                                   "--train" if train else "",
                                                   f"--wait-to-start {wait_to_start}" if wait_to_start!="" else "",
                                                   "--resume" if resume else "")
                                                #    f"--port {port}" if num_gpu > 1 else "")
        if num_gpu > 1:
            bash_command = f"torchrun \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:{port} \
--nnode=1 \
--nproc_per_node={num_gpu} \
{program} {config_str}"
        else:
            bash_command = "python {} --num_gpu {} {}".format(
                program, num_gpu, config_str)
        command_str = "{} {} {}".format(debug_info, prefix, bash_command)
    if test:
        # log.debug(dict_to_string([test, test_only]))
        config_str = "--config {} {} {} {} {}".format(config_file,
                                                   "--resume" if resume else "",
                                                   "--test" if test else "",
                                                   f"--wait-to-start {wait_to_start}" if (wait_to_start!="" and not train) else "",
                                                   "--test_only" if test_only else "")
        bash_command = "python {} --num_gpu {} {}".format(
            program, 1, config_str)
        if command_str is not None:
            command_str = command_str + " && " + "{} {} {}".format(debug_info, prefix, bash_command)
        else:
            command_str = "{} {} {}".format(debug_info, prefix, bash_command)

# --rdzv_backend=c10d \
# --rdzv_endpoint=localhost:{} \

    # debug_info = "export NCCL_DEBUG=INFO; export NCCL_DEBUG_SUBSYS=ALL;"
    # debug_info += "export OMP_NUM_THREADS=4;"
    log.debug("run command: {}".format(command_str))
    assert command_str is not None
    # subprocess.run(command_str, shell=True)
    try:
        result = subprocess.run(
            command_str,
            shell=True,
            check=True,
            text=True,
        )
        log.debug(f"final_output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        log.debug(f"error, returncode: {e.returncode}")
        log.debug(f"stderr: {e.stderr}")
        log.debug(f"stdout: {e.stdout}")
    log.debug("end running command: {}".format(command_str))
    # if test_str is not None:
    #     log.debug("run test_command: {}".format(test_str))
    #     subprocess.run(test_str, shell=True)
    # subprocess.run("{} {} {}".format(prefix, "python {} --config {} --test"), shell=True)
