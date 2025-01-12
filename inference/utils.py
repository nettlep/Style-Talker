import re
from itertools import zip_longest

def merge_repeated_ngrams(text):
    # Split the text into words for processing
    words = text.split()
    
    # Function to find n-grams
    def find_ngrams(input_list, n):
        return zip_longest(*[input_list[i:] for i in range(n)], fillvalue='')
    
    # Loop from 5-grams to unigrams to replace them in the original text
    for n in range(5, 0, -1):
        ngrams = find_ngrams(words, n)
        for ngram in ngrams:
            ngram_str = ' '.join(ngram).strip()
            # Replace only if the ngram occurs more than once consecutively
            if ngram_str and text.count(ngram_str + ' ' + ngram_str) > 0:
                text = text.replace(ngram_str + ' ' + ngram_str, ngram_str)
    return text


def merge_repeated_words(text):
    # Pattern to match a word (with possible apostrophe within) followed by optional punctuation and the same word.
    # Handles case sensitivity by capturing the first occurrence and using it in the replacement.
    pattern = re.compile(r"\b(\w+'?\w*)\b(?:[,.!?;:\s]+)\1\b", re.IGNORECASE)

    # Function to replace matched patterns
    def replace(match):
        # Return the first captured group (the matched word) without any following punctuation or duplicates
        return match.group(1)

    # Continuously apply the replacement until no further replacements can be made
    while True:
        new_text, n = pattern.subn(replace, text)
        if n == 0:  # If no replacements were made, break out of the loop
            break
        text = new_text  # Update the text for the next iteration

    return text


if __name__ == '__main__':


    input_text = "'Yeah, I mean, I mean, I mean, it's, it's, it's a bit of a tough one. I mean, I, I, I, I can, I can understand that, but at the same time, like, it's, it's, it's a, it's a, it's a bit of a weird one. Like, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean'"

    print(input_text)
    print('\n')
    print(merge_repeated_ngrams(input_text))
    print('\n')
    output_text = merge_repeated_words(merge_repeated_ngrams(input_text))
    print(output_text)

