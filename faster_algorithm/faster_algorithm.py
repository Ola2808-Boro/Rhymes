import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import os

my_dictionary = []
directory_path = 'dictionary_no_space'
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        lines = open(file_path,'r', encoding='utf-8').read().split('\n')
        for line in lines:
            my_dictionary.append(line.strip())


file = open("indexed_dictionary.json", "r")
indexed_dictionary = json.load(file)

def get_syllables_count(word):
    vowel_pattern = '[aąeęioóuy]+'
    i_pattern = 'i[aąeęioóuy]'
    diphthong_pattern = '[ae]u'
    vowel_groups = re.findall(vowel_pattern, word)
    count = 0
    for vowel_group in vowel_groups:
        count += len(vowel_group)
        if re.match(i_pattern, vowel_group):
            count -= 1
        if re.match(diphthong_pattern, vowel_group):
            count -= 1
    return count


def get_last_vowels(word):
    word = word.replace('ó', 'u')
    vowel_pattern = '[aąeęioóuy]+'
    i_pattern = 'i[aąeęioóuy]'
    vowel_groups = re.findall(vowel_pattern, word)
    if len(vowel_groups) == 0:
        return ''
    else:
        vowel_group = vowel_groups[-1]
        if re.match(i_pattern, vowel_group):
            return vowel_group[-2:]
        else:
            return vowel_group[-1:]


def get_last_consonants(word, is_last):
    # zakładam, że spółgłoski są na końcu słowa
    consonant_pattern = '[^aąeęioóuy]+'
    consonant_groups = re.findall(consonant_pattern, word)
    if len(consonant_groups) == 0:
        return ''
    else:
        consonant_group = consonant_groups[-1]
        if is_last:
            if consonant_group in ['ż', 'rz']:
                return 'sz'
            elif consonant_group == 'dż':
                return 'cz'
            elif consonant_group == 'b':
                return 'p'
            elif consonant_group == 'g':
                return 'k'
            elif consonant_group == 'dź':
                return 'ć'
            elif consonant_group == 'dz':
                return 'c'
            elif consonant_group == 'w':
                return 'f'
            elif consonant_group == 'd':
                return 't'
            elif consonant_group == 'z':
                return 's'
            elif consonant_group == 'ch':
                return 'h'
            else:
                return consonant_group
        else:
            return consonant_group[-1]


def get_word_ending(word):
    last_vowels = get_last_vowels(word)
    vowel_pattern = '[aąeęioóuy]'
    if re.match(vowel_pattern, word[-1]):
        return get_last_consonants(word, is_last=False) + last_vowels
    else:
        return last_vowels + get_last_consonants(word, is_last=True)


def get_word_key(word):
    syllables_count = get_syllables_count(word)
    ending = get_word_ending(word)
    return str(syllables_count) + '_' + ending


def get_word_beginning(word):
    length = len(word) - len(get_word_ending(word))
    return word[0:length]


def get_indexed_dictionary(dictionary):
    indexed_dictionary = {}
    for word in dictionary:
        key = get_word_key(word)
        if key in indexed_dictionary:
            indexed_dictionary[key].append(word)
        else:
            indexed_dictionary[key] = [word]
    return indexed_dictionary


def get_rhymes_list(word, syllables_counts = -1):
    rhymes_list = []
    ending = get_word_ending(word)
    if syllables_counts == -1:
        syllables_counts = list(range(1,10))
    for syllables_count in syllables_counts:
        key = str(syllables_count) + '_' + ending
        rhymes = indexed_dictionary.get(key, None)
        if rhymes is not None:
            for rhyme in rhymes:
                if rhyme == word:
                    continue
                rhymes_list.append(rhyme)
    return rhymes_list


def get_score(original_word, checked_word):
    original_word_ending = get_word_ending(original_word)
    checked_word_ending = get_word_ending(checked_word)
    score = 0
    if original_word_ending != checked_word_ending:
        final_score = 0
    else:
        original_word_syllables = get_syllables_count(original_word)
        checked_word_syllables = get_syllables_count(checked_word)
        if original_word_syllables == 1:
            score = 1
        else:
            vowel_pattern = '[aąeęioóuy]'
            consonant_pattern = '[^aąeęioóuy]'
            if re.match(vowel_pattern, original_word_ending[-1]):
                score = 0.5
            else:
                score = 0.5

            original_word_beginning = get_word_beginning(original_word)
            checked_word_beginning = get_word_beginning(checked_word)
            original_word_beginning = original_word_beginning.replace('ó', 'u').replace('ch', 'h').replace('au',
                                                                                                           'ał').replace(
                'eu', 'eł')
            checked_word_beginning = checked_word_beginning.replace('ó', 'u').replace('ch', 'h').replace('au',
                                                                                                         'ał').replace(
                'eu', 'eł')

            match = re.search(vowel_pattern, original_word_beginning)
            max_letters = len(original_word_beginning[match.start():])

            same_letters = 0
            for i in range(min(max_letters, len(checked_word_beginning))):
                if original_word_beginning[-1 - i] == checked_word_beginning[-1 - i]:
                    same_letters += 1
                else:
                    break

            score = score + (1 - score) * same_letters / max_letters

        syllables_difference = abs(original_word_syllables - checked_word_syllables)
        final_score = score - 0.05 * syllables_difference
        final_score = max(final_score, 0)
    return round(final_score, 2)


def get_scoreboard(word, syllables_count=-1):
    rhymes = get_rhymes_list(word, syllables_count)
    scoreboard = {rhyme : get_score(word, rhyme) for rhyme in rhymes}
    sorted_scoreboard = sorted(scoreboard.items(), key=lambda x:x[1], reverse=True)
    return sorted_scoreboard


while True:
    word = input("Podaj słowo: ")
    print(get_scoreboard(word))