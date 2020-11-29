def distance_to_doc(matrix_main, comp_doc):
    """
    Returns list of similarity of every doc in matrix_main to comp_doc
    """
    sims = []
    for maindoc in matrix_main.index:
        sim = 1 - spatial.distance.cosine(matrix_main.loc[maindoc], comp_doc)
        sims.append(sim)
    return sims


def doc_len_list(doc_list: list, ignore_less_than: int = 0):
    """list of number of tokens in spacy docs

    Args:
        doc_list (list): list of spacy Docs
        ignore_empty (bool, optional): Include empty documents? Defaults to True.

    Returns:
        float: average number of tokens in spacy docs
    """
    lens = []
    [
        lens.append(len(doc)) if len(doc) >= ignore_less_than else lens.append(None)
        for doc in doc_list
    ]

    return lens