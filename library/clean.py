

def doc_len_list(doc_list:list, ignore_empty:bool=True):
    """list of number of tokens in spacy docs

    Args:
        doc_list (list): list of spacy Docs
        ignore_empty (bool, optional): Include empty documents? Defaults to True.

    Returns:
        float: average number of tokens in spacy docs
    """
    lens = []
    [lens.append(len(doc)) for doc in doc_list if (len(doc) > 0 or ignore_empty==False)]

    return lens






