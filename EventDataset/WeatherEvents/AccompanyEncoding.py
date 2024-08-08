def get_weather_map():
    file2 = open("./EventDataset/WeatherEvents/545110-99999-2017")
    weather_map = {}

    while True:
        line_file2 = file2.readline()
        if not line_file2:
            break
        yearid, monthid, dayid, hourid, precipitation1, _, _, _, wind_speed, _, _, _ = line_file2.split(' ', 12)
        if monthid == '04':
            weather_map[24 * (int(dayid) - 1) + int(hourid)] = str.strip(precipitation1) + "," + str.strip(wind_speed)
        if monthid == '05':
            weather_map[30 * 24 + 24 * (int(dayid) - 1) + int(hourid)] = str.strip(precipitation1) + "," + str.strip(wind_speed)

    file2.close()
    return weather_map