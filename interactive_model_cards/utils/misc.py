def conf_level(val):
    """ Translates probability value into
        a plain english statement """
    # https://www.dni.gov/files/documents/ICD/ICD%20203%20Analytic%20Standards.pdf
    conf = "undefined"
    print(val)
    if val < 0.05:
        conf = "Extremely Low Probability"
    elif val >= 0.05 and val < 0.20:
        conf = "Very Low Probability"
    elif val >= 0.20 and val < 0.45:
        conf = "Low Probability"
    elif val >= 0.45 and val < 0.55:
        conf = "Middling Probability"
    elif val >= 0.55 and val < 0.80:
        conf = "High  Probability"
    elif val >= 0.80 and val < 0.95:
        conf = "Very High Probability"
    elif val >= 0.95:
        conf = "Extremely High Probability"

    return conf
