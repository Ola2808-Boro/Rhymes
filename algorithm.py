import os
import re
class Algorithm():

    def __init__(self) -> None:
        self.words_dictionary = self.load_dictionary()
        self.ending_dictionary =  self.get_ending_dictionary(self.words_dictionary)
        self.indexed_dictionary =  self.get_indexed_dictionary(self.words_dictionary)
    def load_dictionary(self):
        words_dict = []
        directory_path = 'C:/Users/olkab/Desktop/Project Rhyme/Rhymes\dictionary'
        exceptions = []
        for filename in os.listdir(directory_path):
            if filename.endswith("slowa.txt"):
                file_path = os.path.join(directory_path, filename)

                # remove_whitespace(file_path)
                lines = open(file_path, 'r', encoding='utf-8').read().split('\n')
                for line in lines:
                    words_dict.append(line.strip())
        return words_dict



    def get_syllables_count(self,word):
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


    def get_last_vowels(self,word):
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


    def get_last_consonants(self,word):
        consonant_pattern = '[^aąeęioóuy]+'
        consonant_groups = re.findall(consonant_pattern, word)
        if len(consonant_groups) == 0:
            return ''
        else:
            consonant_group = consonant_groups[-1]
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


    def get_word_ending(self,word):
        last_vowels = self.get_last_vowels(word)
        vowel_pattern = '[aąeęioóuy]'
        if re.match(vowel_pattern, word[-1]):
            return last_vowels
        else:
            return last_vowels + self.get_last_consonants(word)


    def get_word_key(self,word):
        syllables_count = self.get_syllables_count(word)
        ending = self.get_word_ending(word)
        return str(syllables_count) + '_' + ending


    def get_word_beginning(self,word):
        vowel_pattern = '[aąeęioóuy]+'
        i_pattern = 'i[aąeęioóuy]'
        vowel_matches = re.finditer(vowel_pattern, word)
        vowel_match = None
        for match in vowel_matches:
            vowel_match = match

        if not vowel_match:
            return ''
        else:
            vowel_group = vowel_match.group()
            if re.match(i_pattern, vowel_group):
                return word[:vowel_match.end() - 2]
            else:
                return word[:vowel_match.end() - 1]


    def get_ending_dictionary(self,dictionary):
        ending_dict = {}
        for word in dictionary:
            key = self.get_word_ending(word)
            if key in ending_dict:
                ending_dict[key].append(word)
            else:
                ending_dict[key] = [word]
        return ending_dict


    def get_indexed_dictionary(self,dictionary):
        indexed_dict = {}
        for word in dictionary:
            key = self.get_word_key(word)
            if key in indexed_dict:
                indexed_dict[key].append(word)
            else:
                indexed_dict[key] = [word]
        return indexed_dict


    def get_score(self,original_word, checked_word):
        original_word_ending = self.get_word_ending(original_word)
        checked_word_ending = self.get_word_ending(checked_word)
        if original_word_ending != checked_word_ending:
            final_score = 0
        else:
            original_word_syllables = self.get_syllables_count(original_word)
            checked_word_syllables = self.get_syllables_count(checked_word)
            if original_word_syllables == 1:
                score = 1
            else:
                vowel_pattern = '[aąeęioóuy]'
                if re.match(vowel_pattern, original_word_ending[-1]):
                    score = 0.4
                else:
                    score = 0.7

                original_word_beginning = self.get_word_beginning(original_word)
                checked_word_beginning = self.get_word_beginning(checked_word)
                original_word_beginning = (original_word_beginning.
                                        replace('ó', 'u').
                                        replace('ch', 'h').
                                        replace('au', 'ał').
                                        replace('eu', 'eł'))
                checked_word_beginning = (checked_word_beginning.
                                        replace('ó', 'u').
                                        replace('ch', 'h').
                                        replace('au', 'ał').
                                        replace('eu', 'eł'))

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
            final_score = score * (1 - 0.05 * syllables_difference)
            final_score = max(final_score, 0)
        return round(final_score, 2)


    def get_rhymes_list(self,word, syllables_count):
        rhymes_list = []
        ending =self.get_word_ending(word)
        if syllables_count == -1:
            key = ending
            rhymes =self.ending_dictionary.get(key, None)
        else:
            key = str(syllables_count) + '_' + ending
            rhymes = self.indexed_dictionary.get(key, None)
        if rhymes is not None:
            for rhyme in rhymes:
                if rhyme == word:
                    continue
                rhymes_list.append(rhyme)
        return rhymes_list


    def get_scoreboard(self,word, syllables_count, quality=0):
        rhymes = self.get_rhymes_list(word, syllables_count)
        scoreboard = [{"word": rhyme,
                    "quality": self.get_score(word, rhyme)
                    # "syllables_count": get_syllables_count(rhyme)
                    }
                    for rhyme in rhymes
                    if  self.get_score(word, rhyme) >= quality]
        sorted_scoreboard = sorted(scoreboard, key=lambda x: (-x["quality"], x["word"]))
        return sorted_scoreboard
    

