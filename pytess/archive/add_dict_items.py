

def add_dict_items(dict_object, **kwargs):
    """
    get the added dictionary with several key-value pairs and combine with the input original dictionary
    :param dict_object:
    :param kwargs:
    :return:
    """
    n_argin_var = 1
    # print(kwargs.keys()
    dict_object.update(kwargs)

    return dict_object
