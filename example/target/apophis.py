import satastro

horizons_args_vec = {'COMMAND':'Apophis',       # Target
                     'OBJ_DATA':'YES',
                     'MAKE_EPHEM':'YES',
                     'TABLE_TYPE':'VECTORS',
                     'CENTER':'geo',
                     'REF_PLANE':'FRAME',
                     'START_TIME':'2029-04-11', # Start date of ephemerides
                     'STOP_TIME':'2029-04-17',  # End date of ephemerides
                     'STEP_SIZE':'1 MINUTES',   # Time step of ephemerides
                     'TIME_TYPE':'UT',
                     'CSV_FORMAT':'YES',}
apophis_horizons_vec_text = satastro.request_horizons(**horizons_args_vec)
apophis_horizons_vec = satastro.horizons_to_dataframe(apophis_horizons_vec_text)
apophis_teme = satastro.horizons_to_TEME(apophis_horizons_vec)

horizons_args_mag = {'MAKE_EPHEM':'YES',
                     'COMMAND':'Apophis',
                     'EPHEM_TYPE':'OBSERVER',
                     'CENTER':'500@399',
                     'START_TIME':'2029-04-11',
                     'STOP_TIME':'2029-04-17',
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
apophis_horizons_mag_text = satastro.request_horizons(**horizons_args_mag)
apophis_mag = satastro.horizons_to_dataframe(apophis_horizons_mag_text)