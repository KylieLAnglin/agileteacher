

def doc_len_list(doc_list:list, ignore_less_than:int=0):
    """list of number of tokens in spacy docs

    Args:
        doc_list (list): list of spacy Docs
        ignore_empty (bool, optional): Include empty documents? Defaults to True.

    Returns:
        float: average number of tokens in spacy docs
    """
    lens = []
    [lens.append(len(doc)) if len(doc) >= ignore_less_than else lens.append(None) for doc in doc_list ]

    return lens






