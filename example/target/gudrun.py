import satastro

horizons_args_vec = {'COMMAND':'Gudrun',       # Target
                     'OBJ_DATA':'YES',
                     'MAKE_EPHEM':'YES',
                     'TABLE_TYPE':'VECTORS',
                     'CENTER':'geo',
                     'REF_PLANE':'FRAME',
                     'START_TIME':'2029-03-20', # Start date of ephemerides
                     'STOP_TIME':'2029-03-23',  # End date of ephemerides
                     'STEP_SIZE':'1 MINUTES',   # Time step of ephemerides
                     'TIME_TYPE':'UT',
                     'CSV_FORMAT':'YES',}
asteroid_horizons_vec_text = satastro.request_horizons(**horizons_args_vec)
asteroid_horizons_vec = satastro.horizons_to_dataframe(asteroid_horizons_vec_text)
asteroid_teme = satastro.horizons_to_TEME(asteroid_horizons_vec)

horizons_args_mag = {'MAKE_EPHEM':'YES',
                     'COMMAND':'Gudrun',
                     'EPHEM_TYPE':'OBSERVER',
                     'CENTER':'500@399',
                     'START_TIME':'2029-03-20',
                     'STOP_TIME':'2029-03-23',
                     'STEP_SIZE':'1 MINUTES',
                     'TIME_TYPE':'UT',
                     'QUANTITIES':'1,9,10,13,20,23,24,25,47,48',
                     'REF_SYSTEM':'ICRF',
                     'CAL_FORMAT':'CAL',
                     'CAL_TYPE':'M',
                     'TIME_DIGITS':'SECONDS',
                     'ANG_FORMAT':'DEG',
                     'APPARENT':'AIRLESS',
                     'RANGE_UNITS':'KM',
                     'SUPPRESS_RANGE_RATE':'NO',
                     'SKIP_DAYLT':'NO',
                     'SOLAR_ELONG':'0,180',
                     'EXTRA_PREC':'NO',
                     'R_T_S_ONLY':'NO',
                     'CSV_FORMAT':'YES',
                     'OBJ_DATA':'YES',}
asteroid_horizons_mag_text = satastro.request_horizons(**horizons_args_mag)
asteroid_mag = satastro.horizons_to_dataframe(asteroid_horizons_mag_text)