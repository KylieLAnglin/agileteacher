def distance_to_doc(matrix_main, comp_doc):
    """
    Returns list of similarity of every doc in matrix_main to comp_doc
    """
    sims = []
    for maindoc in matrix_main.index:
        sim = 1 - spatial.distance.cosine(matrix_main.loc[maindoc], comp_doc)
        sims.append(sim)
    return sims