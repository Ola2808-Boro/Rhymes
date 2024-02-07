from flask import Flask, request, jsonify, render_template
from flaskwebgui import FlaskUI
from algorithm import Algorithm

app = Flask(__name__, template_folder='static', static_url_path='', static_folder='static')
algorithm=Algorithm()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/rhymes', methods=['GET'])
def get_rhymes():
    # try:
    #     # json_file = request.args.to_dict() # e.g. {'word': 'test', 'quality': '0.5', 'syllables_count': '13'}
    #     # return [{'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13},{'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13},{'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13},{'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13},{'word': 'test', 'quality': 0.5, 'syllables_count': 13}]
    #     # message = request.get_json(force=True)
    #     message = request.args.to_dict()
    #     word = message['word']
    #     quality = message['quality']
    #     syllables_count = message['syllables_count']
    #     scoreboard = get_scoreboard(word, syllables_count=syllables_count)
    #     return scoreboard
    #     # response = jsonify(scoreboard=scoreboard)
    #     # response.headers.add('Access-Control-Allow-Origin', '*')
    #     # return response
    try:
        json_file = request.args.to_dict() # e.g. {'word': 'test', 'quality': '0.5', 'syllables_count': '13'}
        # return [{'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13},{'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13},{'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13},{'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13}, {'word': 'test', 'quality': 0.5, 'syllables_count': 13},{'word': 'test', 'quality': 0.5, 'syllables_count': 13}]
        word = json_file["word"]
        if "syllables_count" in json_file:
            syllables_count = int(json_file["syllables_count"])
        else:
            syllables_count = -1
        if "quality" in json_file:
            quality = float(json_file["quality"])
        else:
            quality = 0
        return algorithm.get_scoreboard(word, syllables_count, quality)
        # message = request.get_json(force=True)

    except Exception as err:
        return jsonify({'message': err, 'status_id': 2})

# def load_dictionary():
#     words_dict = []
#     directory_path = 'dictionary'
#     exceptions = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith("slowa.txt"):
#             file_path = os.path.join(directory_path, filename)

#             # remove_whitespace(file_path)
#             lines = open(file_path, 'r', encoding='utf-8').read().split('\n')
#             for line in lines:
#                 words_dict.append(line.strip())
#     return words_dict



# def get_syllables_count(word):
#     vowel_pattern = '[aąeęioóuy]+'
#     i_pattern = 'i[aąeęioóuy]'
#     diphthong_pattern = '[ae]u'
#     vowel_groups = re.findall(vowel_pattern, word)
#     count = 0
#     for vowel_group in vowel_groups:
#         count += len(vowel_group)
#         if re.match(i_pattern, vowel_group):
#             count -= 1
#         if re.match(diphthong_pattern, vowel_group):
#             count -= 1
#     return count


# def get_last_vowels(word):
#     word = word.replace('ó', 'u')
#     vowel_pattern = '[aąeęioóuy]+'
#     i_pattern = 'i[aąeęioóuy]'
#     vowel_groups = re.findall(vowel_pattern, word)
#     if len(vowel_groups) == 0:
#         return ''
#     else:
#         vowel_group = vowel_groups[-1]
#         if re.match(i_pattern, vowel_group):
#             return vowel_group[-2:]
#         else:
#             return vowel_group[-1:]


# def get_last_consonants(word):
#     consonant_pattern = '[^aąeęioóuy]+'
#     consonant_groups = re.findall(consonant_pattern, word)
#     if len(consonant_groups) == 0:
#         return ''
#     else:
#         consonant_group = consonant_groups[-1]
#         if consonant_group in ['ż', 'rz']:
#             return 'sz'
#         elif consonant_group == 'dż':
#             return 'cz'
#         elif consonant_group == 'b':
#             return 'p'
#         elif consonant_group == 'g':
#             return 'k'
#         elif consonant_group == 'dź':
#             return 'ć'
#         elif consonant_group == 'dz':
#             return 'c'
#         elif consonant_group == 'w':
#             return 'f'
#         elif consonant_group == 'd':
#             return 't'
#         elif consonant_group == 'z':
#             return 's'
#         elif consonant_group == 'ch':
#             return 'h'
#         else:
#             return consonant_group


# def get_word_ending(word):
#     last_vowels = get_last_vowels(word)
#     vowel_pattern = '[aąeęioóuy]'
#     if re.match(vowel_pattern, word[-1]):
#         return last_vowels
#     else:
#         return last_vowels + get_last_consonants(word)


# def get_word_key(word):
#     syllables_count = get_syllables_count(word)
#     ending = get_word_ending(word)
#     return str(syllables_count) + '_' + ending


# def get_word_beginning(word):
#     vowel_pattern = '[aąeęioóuy]+'
#     i_pattern = 'i[aąeęioóuy]'
#     vowel_matches = re.finditer(vowel_pattern, word)
#     vowel_match = None
#     for match in vowel_matches:
#         vowel_match = match

#     if not vowel_match:
#         return ''
#     else:
#         vowel_group = vowel_match.group()
#         if re.match(i_pattern, vowel_group):
#             return word[:vowel_match.end() - 2]
#         else:
#             return word[:vowel_match.end() - 1]


# def get_ending_dictionary(dictionary):
#     ending_dict = {}
#     for word in dictionary:
#         key = get_word_ending(word)
#         if key in ending_dict:
#             ending_dict[key].append(word)
#         else:
#             ending_dict[key] = [word]
#     return ending_dict


# def get_indexed_dictionary(dictionary):
#     indexed_dict = {}
#     for word in dictionary:
#         key = get_word_key(word)
#         if key in indexed_dict:
#             indexed_dict[key].append(word)
#         else:
#             indexed_dict[key] = [word]
#     return indexed_dict


# def get_score(original_word, checked_word):
#     original_word_ending = get_word_ending(original_word)
#     checked_word_ending = get_word_ending(checked_word)
#     if original_word_ending != checked_word_ending:
#         final_score = 0
#     else:
#         original_word_syllables = get_syllables_count(original_word)
#         checked_word_syllables = get_syllables_count(checked_word)
#         if original_word_syllables == 1:
#             score = 1
#         else:
#             vowel_pattern = '[aąeęioóuy]'
#             if re.match(vowel_pattern, original_word_ending[-1]):
#                 score = 0.4
#             else:
#                 score = 0.7

#             original_word_beginning = get_word_beginning(original_word)
#             checked_word_beginning = get_word_beginning(checked_word)
#             original_word_beginning = (original_word_beginning.
#                                        replace('ó', 'u').
#                                        replace('ch', 'h').
#                                        replace('au', 'ał').
#                                        replace('eu', 'eł'))
#             checked_word_beginning = (checked_word_beginning.
#                                       replace('ó', 'u').
#                                       replace('ch', 'h').
#                                       replace('au', 'ał').
#                                       replace('eu', 'eł'))

#             match = re.search(vowel_pattern, original_word_beginning)
#             max_letters = len(original_word_beginning[match.start():])

#             same_letters = 0
#             for i in range(min(max_letters, len(checked_word_beginning))):
#                 if original_word_beginning[-1 - i] == checked_word_beginning[-1 - i]:
#                     same_letters += 1
#                 else:
#                     break

#             score = score + (1 - score) * same_letters / max_letters

#         syllables_difference = abs(original_word_syllables - checked_word_syllables)
#         final_score = score * (1 - 0.05 * syllables_difference)
#         final_score = max(final_score, 0)
#     return round(final_score, 2)


# def get_rhymes_list(word, syllables_count):
#     rhymes_list = []
#     ending = get_word_ending(word)
#     if syllables_count == -1:
#         key = ending
#         rhymes = ending_dictionary.get(key, None)
#     else:
#         key = str(syllables_count) + '_' + ending
#         rhymes = indexed_dictionary.get(key, None)
#     if rhymes is not None:
#         for rhyme in rhymes:
#             if rhyme == word:
#                 continue
#             rhymes_list.append(rhyme)
#     return rhymes_list


# def get_scoreboard(word, syllables_count, quality=0):
#     rhymes = get_rhymes_list(word, syllables_count)
#     scoreboard = [{"word": rhyme,
#                    "quality": get_score(word, rhyme)
#                    # "syllables_count": get_syllables_count(rhyme)
#                    }
#                   for rhyme in rhymes
#                   if get_score(word, rhyme) >= quality]
#     sorted_scoreboard = sorted(scoreboard, key=lambda x: (-x["quality"], x["word"]))
#     return sorted_scoreboard


if __name__ == '__main__':
    words_dictionary = load_dictionary()
    ending_dictionary = get_ending_dictionary(words_dictionary)
    indexed_dictionary = get_indexed_dictionary(words_dictionary)
    FlaskUI(app=app, server="flask", port=5000).run()

