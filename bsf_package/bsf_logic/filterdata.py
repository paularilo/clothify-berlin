def filtercategory(data, choice_shop):
    mask = data.final_categories.apply(lambda x: any(item for item in \
        choice_shop if item in x)) # filter df if any category matching
    df = data[mask]
    return df

def filterdistrict(data):
    pass
