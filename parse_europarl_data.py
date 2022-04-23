def read_tokens_file(token_path, num_sentences):
    """
    Reads in a token file path and returns the data, limited to the number of sentences.
    :param token_path: path to token file.
    :param num_sentences: number of sentences we want to read.
    :return:

    (pseudo-tokenized sentences in language 1, pseudo-tokenized sentences in language 2)

    """

    tokens_lang_1 = []
    tokens_lang_2 = []

    with open(token_path, "r", encoding='utf-8') as file:
        """
        The content should be of form
        
        sentence_in_language_1 ||| sentence_in_language_2
        """

        for idx, line in enumerate(file):

            # if we got enough data
            if idx >= num_sentences:
                break

            # split into the 2 languages and then split further into "tokens" (not really tokens, but naming works for now)
            s1, s2 = line.split("|||")
            tokens_s1 = s1.split()
            tokens_s2 = s2.split()

            tokens_lang_1.append(tokens_s1)
            tokens_lang_2.append(tokens_s2)

    return tokens_lang_1, tokens_lang_2


def read_alignments_file(alignment_path, num_sentences):
    """
    Reads in an alignment file path and returns the data, limited to the number of sentences.
    :param alignment_path: path to the file containing the alignments
    :param num_sentences: number of sentences we are interested in
    :return: list of alignments per sentence
    """

    with open(alignment_path, "r", encoding='utf-8') as align_file:
        """
        Content should be of form
        
        0-0 1-1 2-3 etc.
        
        """
        alignments = []

        for idx, line in enumerate(align_file):
            # if we got enough data
            if idx >= num_sentences:
                break

            pairs = [p.split("-") for p in line.split()]
            alignments.append(pairs)

    return alignments


def create_parallel_sentences(token_files, alignment_files, num_sentences=15):
    """
    Takes in an array of token files and array of alignment files and constructs the data to further fine-tune BERT.

    :param token_files: Array
        Array of file paths to files containing cleaned tokenized data.
    :param alignment_files:
        Array of file paths to alignment files.
    :return: array 3-tuples, one tuple for each token - alignment pair
        ( array of tokenized sentences in language 1,
        array of tokenized sentences in language 2,
        array of sentence token alignments
        )
    """

    data = []

    if len(token_files) is not len(alignment_files):
        raise Exception("Unequal number of token and alignment files.")

    for token_path, align_path in zip(token_files, alignment_files):
        tokens_1, tokens_2 = read_tokens_file(token_path, num_sentences)
        alignments = read_alignments_file(align_path, num_sentences)
        data.append((tokens_1, tokens_2, alignments))

    return data