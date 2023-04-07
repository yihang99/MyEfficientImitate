import subprocess
import os
import time

import ipdb

import logging

log = logging.getLogger(__name__)


def kill_carla():
    kill_process = subprocess.Popen('killall -9 -r CarlaUE4-Linux', shell=True)
    kill_process.wait()
    time.sleep(1)
    log.info("Kill Carla Servers!")


class CarlaServerManager:
    def __init__(self, carla_sh_str, port=4000, configs=None, t_sleep=5):
        self._carla_sh_str = carla_sh_str
        # self._root_save_dir = root_save_dir
        self._t_sleep = t_sleep
        self.env_configs = []

        if configs is None:
            cfg = {
                'gpu': 0,
                'port': port,
            }
            self.env_configs.append(cfg)
        else:
            for cfg in configs:
                cfg['port'] = port
                self.env_configs.append(cfg)
                port += 5

    def start(self, gpu=None, kill_server=False):
        if kill_server:
            kill_carla()
        for cfg in self.env_configs:
            if gpu is None:
                gpu = cfg["gpu"]
            cmd = f'CUDA_VISIBLE_DEVICES={gpu} bash {self._carla_sh_str} ' \
                  f'-fps=10 -quality-level=Epic -world-port={cfg["port"]}' \
                  f' -resx=800 -resy=600'
            cmd = f'echo "{cmd}" && {cmd}'
            # cmd = f'bash ~/carla_suite/carla/CarlaUE4.sh -quality-level=Epic -world-port=1001 -resx=800 -resy=600'
            log.info(cmd)
            server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        time.sleep(self._t_sleep)

    def stop(self):
        kill_carla()
        time.sleep(5)
        log.info(f"Kill Carla Servers!")
