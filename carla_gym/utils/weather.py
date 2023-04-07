import carla
import numpy as np

'''
This file contains the definition of WeatherHandler
'''
WEATHERS = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.ClearSunset,

    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.CloudySunset,

    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.WetSunset,

    carla.WeatherParameters.MidRainyNoon,
    carla.WeatherParameters.MidRainSunset,

    carla.WeatherParameters.WetCloudyNoon,
    carla.WeatherParameters.WetCloudySunset,

    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.HardRainSunset,

    carla.WeatherParameters.SoftRainNoon,
    carla.WeatherParameters.SoftRainSunset,
]